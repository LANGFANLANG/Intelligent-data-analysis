"""
Schema 序列化模块
─────────────────
将数据库 Schema 字典转换为 LLM 友好的文本格式，
用于注入 Agent 提示词上下文。

输出格式:
  表: orders
  ------------------------------------------------------------
  行数: ~50,000
  列名                      类型                 可空    键       说明

    order_id               BIGINT               NO     PK       订单ID
    merchant_id            BIGINT               NO              商家ID
    ...

设计要点:
  - 限制每表最多 40 列（MAX_SCHEMA_COLS），防止单个宽表占满上下文
  - 总输出限制 4000 字符（MAX_SCHEMA_TEXT_LENGTH），超出截断并标注
  - 处理 NULL 注释、超长类型名等边界情况
"""
MAX_SCHEMA_COLS = 40
MAX_SCHEMA_TEXT_LENGTH = 4000


def serialize_table_schema(table_name: str, columns: list[dict],
                           row_count: int | None = None) -> str:
    """将单张表的结构信息序列化为可读文本

    输出包含: 表头分隔线、行数、列名/类型/可空/主键/注释的格式化表格。

    Args:
        table_name: 表名
        columns:    列信息列表，每项含 name/type/nullable/comment/is_primary_key
        row_count:  大致行数（可选）

    Returns:
        格式化的多行文本
    """
    lines = [f"表: {table_name}", "-" * 60]
    if row_count is not None:
        lines.append(f"行数: ~{row_count:,}")
    lines.append(f"{'列名':<25} {'类型':<20} {'可空':<6} {'键':<8} 说明")
    lines.append("")

    for col in columns[:MAX_SCHEMA_COLS]:
        pk_flag = "PK" if col.get("is_primary_key") else ""
        nullable = "YES" if col.get("nullable") else "NO"
        # 使用 or "" 防御 None 值（SQLAlchemy 可能返回 None 的 comment）
        comment = (col.get("comment") or "")[:30]
        type_str = col.get("type") or ""
        # 超长类型名截断，如 VARCHAR(255) COLLATE "utf8mb4_unicode_ci"
        if len(type_str) > 18:
            type_str = type_str[:15] + "..."
        lines.append(
            f"  {col['name']:<23} {type_str:<20} {nullable:<6} {pk_flag:<8} {comment}"
        )

    if len(columns) > MAX_SCHEMA_COLS:
        lines.append(f"  ... 还有 {len(columns) - MAX_SCHEMA_COLS} 列")

    return "\n".join(lines)


def serialize_schema(schema: dict) -> str:
    """将完整数据库 Schema 序列化为 LLM 友好的文本

    所有表按顺序拼接，表之间空行分隔。
    输出总长度超过限制时截断并标注。

    Args:
        schema: {table_name: {"columns": [...], "row_count": N}, ...}

    Returns:
        完整 Schema 文本；空 Schema 返回 "暂无可用表"
    """
    if not schema:
        return "暂无可用表"

    parts = []
    for table_name, info in schema.items():
        cols = info.get("columns", [])
        row_count = info.get("row_count")
        # 标记发现失败的表
        if "error" in info:
            parts.append(f"表: {table_name} (无法读取: {info['error']})")
            continue
        parts.append(serialize_table_schema(table_name, cols, row_count))

    full = "\n\n".join(parts)
    # 防止 Schema 文本过长超出 LLM 上下文窗口
    if len(full) > MAX_SCHEMA_TEXT_LENGTH:
        full = full[:MAX_SCHEMA_TEXT_LENGTH] + "\n\n... (schema 过长已截断)"

    return full
