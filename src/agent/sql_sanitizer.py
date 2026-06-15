"""
SQL 安全校验模块
────────────────
对用户/LLM 生成的 SQL 进行多层安全校验，确保只执行只读查询。

防护策略:
  第1层: sqlglot AST 解析 → 检查语句类型（仅允许 SELECT / CTE）
  第2层: 关键字检查 → 拦截 INTO OUTFILE / DUMPFILE 等危险操作

禁止的 SQL 操作类型:
  Insert, Update, Delete, Drop, Alter, Truncate, Create,
  Replace, Grant, Revoke, Call, Execute, Load, Set, Comment

依赖:
  sqlglot (>=25.0.0): 纯 Python SQL 解析器，生成 AST 而非简单分词

用法:
    from src.agent.sql_sanitizer import validate_readonly_sql

    ok, err = validate_readonly_sql(user_sql)
    if not ok:
        raise ValueError(f"SQL 不安全: {err}")
"""
import sqlglot
from sqlglot import exp

# ── 禁止的操作类型 ──
# 基于 sqlglot.exp 的表达式类名，匹配 AST 节点类型
FORBIDDEN_TYPES = {
    "Insert", "Update", "Delete", "Drop", "Alter",
    "Truncate", "Create", "Replace", "Grant", "Revoke",
    "Call", "Execute", "Load", "Set", "Comment",
}


def validate_readonly_sql(sql: str) -> tuple[bool, str]:
    """校验 SQL 是否为安全的只读查询

    校验流程:
      1. 清理末尾分号
      2. sqlglot 解析为 AST
      3. 遍历所有顶层语句，检查类型是否在禁止列表中
      4. 特殊处理 CTE（允许，但内部必须包含 SELECT）
      5. 关键字扫描拦截 INTO OUTFILE / DUMPFILE 等

    Args:
        sql: 原始 SQL 语句

    Returns:
        (是否安全, 错误描述)
        - (True, "") 表示通过校验
        - (False, "原因") 表示被拦截
    """
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return False, "SQL 语句为空"

    # sqlglot 解析为语句列表，每个语句是一个 AST 节点
    try:
        statements = sqlglot.parse(sql, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception as e:
        return False, f"SQL 语法解析失败: {e}"

    if not statements:
        return False, "无法解析 SQL 语句"

    # 遍历所有顶层语句进行安全检查
    for stmt in statements:
        if stmt is None:
            continue

        stmt_type = type(stmt).__name__

        if stmt_type in FORBIDDEN_TYPES:
            return False, f"禁止的SQL操作: {stmt_type}"

        if stmt_type == "Select":
            continue

        # CTE (WITH ... SELECT ...) 是安全的，但内部必须包含 SELECT
        if stmt_type == "CTE":
            cte = stmt.find(exp.Select)
            if cte is None:
                return False, "CTE 必须包含 SELECT 语句"
            continue

        return False, f"不支持的SQL语句类型: {stmt_type}"

    # 关键字扫描：拦截 AST 无法覆盖的危险操作
    keywords_lower = sql.lower()
    dangerous_keywords = [
        "into outfile",     # 导出文件
        "into dumpfile",    # 二进制导出
        "load_file",        # 读取服务器文件
        "load data",        # 批量导入
    ]
    for kw in dangerous_keywords:
        if kw in keywords_lower:
            return False, f"禁止的危险操作: {kw}"

    return True, ""
