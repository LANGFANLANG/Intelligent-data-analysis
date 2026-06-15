"""
MySQL 数据库连接器
──────────────────
基于 SQLAlchemy + pymysql 的 MySQL 连接器实现。

安全策略 (多层防护):
  第1层: MySQL 只读用户权限 (GRANT SELECT ONLY)
  第2层: 每次查询前 SET SESSION TRANSACTION READ ONLY
  第3层: sql_sanitizer.py 的 AST 级 SQL 校验
  第4层: 自动追加 LIMIT + 超时控制

连接池配置:
  - pool_size=3: 连接池大小（单用户场景足够）
  - pool_recycle=1800: 30分钟自动回收连接
  - pool_pre_ping=True: 使用前检测连接有效性

用法:
    from src.connectors.mysql_connector import create_connector, get_connector

    connector = create_connector()        # 首次调用时创建
    if connector and connector.is_connected:
        df = connector.execute_query("SELECT * FROM orders LIMIT 10")
"""
import threading
import pandas as pd
from sqlalchemy import create_engine, text, Engine, inspect
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from src.connectors.base import AbstractConnector
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_CHARSET
from src.logger import get_logger

_log = get_logger("mysql_connector")

# ── 全局单例连接器 ──
# 因为当前是单用户共享连接模式，使用模块级全局变量维护连接状态
_connector: "MySQLConnector | None" = None
_lock = threading.Lock()


class MySQLConnector(AbstractConnector):
    """MySQL 数据库连接器

    封装 SQLAlchemy 引擎的创建、查询执行和元数据发现。
    自动创建连接池，每次查询强制只读事务。

    Attributes:
        _host:     数据库主机地址
        _port:     数据库端口
        _database: 数据库名
        _user:     用户名
        _password: 密码
        _charset:  字符集
        _engine:   SQLAlchemy Engine 实例
        _connected: 连接状态
    """

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str, charset: str = "utf8mb4"):
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._charset = charset
        self._engine: Engine | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._engine is not None

    def connect(self) -> bool:
        """建立 MySQL 连接并创建连接池

        使用 pymysql 驱动，连接 URL 格式:
          mysql+pymysql://user:password@host:port/database?charset=utf8mb4

        Returns:
            True 表示连接成功
        """
        url = (
            f"mysql+pymysql://{self._user}:{self._password}"
            f"@{self._host}:{self._port}/{self._database}"
            f"?charset={self._charset}"
        )
        try:
            self._engine = create_engine(
                url,
                pool_size=3,
                pool_recycle=1800,      # 30分钟回收，防止 MySQL wait_timeout 断连
                pool_pre_ping=True,     # 使用前 ping 检测连接
                connect_args={"connect_timeout": 10},
            )
            # 验证连接可用
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._connected = True
            _log.info("MySQL 连接成功: %s:%d/%s", self._host, self._port, self._database)
            return True
        except OperationalError as e:
            _log.error("MySQL 连接失败: %s", e)
            self._connected = False
            return False
        except Exception as e:
            _log.error("MySQL 连接异常: %s", e)
            self._connected = False
            return False

    def execute_query(self, sql: str, max_rows: int = 1000, timeout: int = 30) -> pd.DataFrame:
        """执行只读 SQL 查询

        安全措施:
          1. 强制检查 SQL 以 SELECT 开头
          2. 缺失 LIMIT 时自动追加（防止全表拉取）
          3. 设置事务只读 + 查询超时
          4. 结果转为 DataFrame 返回

        Args:
            sql:      SELECT 查询语句
            max_rows: 最大返回行数
            timeout:  查询超时（秒）

        Returns:
            查询结果 DataFrame

        Raises:
            RuntimeError: 数据库未连接
            ValueError:   非 SELECT 查询
        """
        if not self._engine:
            raise RuntimeError("数据库未连接")

        # 清理 SQL 结尾分号
        sql = sql.strip().rstrip(";").strip()
        if not sql.lower().startswith("select"):
            raise ValueError("仅允许执行 SELECT 查询")

        # 自动追加 LIMIT（如果用户未指定），防止巨大结果集
        sql_with_limit = sql
        lower_sql = sql.lower()
        if "limit" not in lower_sql:
            sql_with_limit = f"SELECT * FROM ({sql}) AS _dq LIMIT {max_rows}"
        else:
            sql_with_limit = sql

        try:
            with self._engine.connect() as conn:
                # 设置只读事务模式（第2层防护）
                conn = conn.execution_options(
                    isolation_level="READ UNCOMMITTED",
                )
                conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
                # MySQL 5.7+ 支持 MAX_EXECUTION_TIME，防止慢查询
                conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={timeout * 1000}"))

                result = conn.execute(text(sql_with_limit))
                rows = result.fetchall()
                columns = list(result.keys())
                df = pd.DataFrame(rows, columns=columns)
                _log.info("SQL 执行成功, 返回 %d 行 %d 列", len(df), len(df.columns))
                return df
        except SQLAlchemyError as e:
            _log.error("SQL 执行失败: %s", e)
            raise

    def discover_tables(self) -> list[str]:
        """获取数据库中所有用户表名

        使用 SQLAlchemy inspect 查询 INFORMATION_SCHEMA。
        返回的是表名列表（不包含视图）。

        Returns:
            表名列表
        """
        if not self._engine:
            return []
        insp = inspect(self._engine)
        tables = insp.get_table_names()
        _log.info("发现 %d 张表", len(tables))
        return tables

    def describe_table(self, table_name: str) -> list[dict]:
        """获取单表结构的完整信息

        通过 SQLAlchemy inspect 获取:
          - 列名 (name)
          - 数据类型 (type)
          - 是否可空 (nullable)
          - 默认值 (default)
          - 注释 (comment)
          - 是否主键 (is_primary_key，经 PK 约束推断)
          - 外键信息 (fk_info，暂未返回给上层)

        Args:
            table_name: 目标表名

        Returns:
            列信息字典列表
        """
        if not self._engine:
            return []
        insp = inspect(self._engine)
        columns = []
        for col in insp.get_columns(table_name):
            # 使用 or "" 兜底：SQLAlchemy 返回的 comment/default 可能为 None
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "default": str(col.get("default") or ""),
                "comment": col.get("comment") or "",
            })

        # 获取主键列名集合
        pk_cols = set()
        try:
            pk = insp.get_pk_constraint(table_name)
            pk_cols = set(pk.get("constrained_columns", []))
        except Exception:
            pass

        # 标记主键列
        for col in columns:
            col["is_primary_key"] = col["name"] in pk_cols

        return columns

    def get_table_row_count(self, table_name: str) -> int | None:
        """获取表的大致行数

        优先从 INFORMATION_SCHEMA.TABLES 读取近似值（瞬时），
        失败时降级为 SELECT COUNT(*)（可能较慢）。

        Args:
            table_name: 表名

        Returns:
            行数估算值；获取失败返回 None
        """
        if not self._engine:
            return None
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
                # 从系统表快速获取近似行数
                result = conn.execute(
                    text("SELECT TABLE_ROWS FROM information_schema.TABLES "
                         "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl"),
                    {"db": self._database, "tbl": table_name},
                )
                row = result.fetchone()
                return row[0] if row else None
        except Exception:
            pass

        # 降级方案: 直接 COUNT(*)
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM `{table_name}`")
                )
                return result.fetchone()[0]
        except Exception:
            return None

    def test_connection(self) -> tuple[bool, str]:
        """测试连接是否可用

        Returns:
            (是否可用, 错误信息)
        """
        if not self._engine:
            return False, "数据库引擎未初始化"
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, ""
        except OperationalError as e:
            return False, f"连接失败: {e}"
        except Exception as e:
            return False, str(e)

    def close(self):
        """关闭连接池，释放引擎资源"""
        if self._engine:
            self._engine.dispose()
            self._connected = False
            _log.info("MySQL 连接已关闭")


def get_connector() -> MySQLConnector | None:
    """获取当前全局连接器实例

    Returns:
        MySQLConnector 实例或 None（未初始化时）
    """
    return _connector


def create_connector() -> MySQLConnector | None:
    """创建或获取数据库连接器（单例模式）

    首次调用时根据 .env 中的 DB_* 配置创建连接。
    后续调用复用已有连接（如果仍然有效）。

    Returns:
        MySQLConnector 实例；配置不完整或连接失败时返回 None
    """
    global _connector
    with _lock:
        # 已有有效连接则直接复用
        if _connector is not None and _connector.is_connected:
            return _connector

        # 检查配置完整性
        if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
            _log.warning("MySQL 配置不完整, 跳过数据库连接")
            return None

        connector = MySQLConnector(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            charset=DB_CHARSET,
        )
        if connector.connect():
            _connector = connector
            return connector
        return None


def close_connector():
    """关闭并销毁全局连接器"""
    global _connector
    with _lock:
        if _connector:
            _connector.close()
            _connector = None
