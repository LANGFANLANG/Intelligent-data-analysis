"""
数据库 Schema 发现模块
──────────────────────
通过连接器自动发现数据库中的所有表及其结构信息。

功能:
  discover_schema()   - 全量发现: 表名 → 列信息 → 行数
  get_cached_schema() - 带 TTL 缓存的获取（推荐使用）
  refresh_schema()    - 强制刷新缓存

缓存机制:
  - 缓存存储为模块级全局变量（_cache / _cache_time）
  - TTL 通过 SCHEMA_CACHE_TTL_MINUTES 环境变量配置
  - 线程安全: 使用 threading.Lock 保护读写

应用层（UI / Agent）应使用 get_cached_schema(),
只在用户点击"刷新 Schema"时才调用 refresh_schema()。
"""
import time
import threading
from src.connectors import get_connector
from src.config import SCHEMA_CACHE_TTL_MINUTES
from src.logger import get_logger

_log = get_logger("schema")

# ── Schema 缓存 ──
_cache: dict | None = None       # {table_name: {"columns": [...], "row_count": N}}
_cache_time: float = 0            # 最后缓存时间戳
_cache_lock = threading.Lock()    # 线程安全锁


def discover_schema() -> dict:
    """全量发现数据库中所有表的结构信息

    遍历所有表，逐表获取列信息和行数估算。
    单表获取失败不影响其他表的发现。

    Returns:
        {table_name: {"columns": [...], "row_count": N | None}, ...}
        连接不可用或数据库为空时返回 {}
    """
    connector = get_connector()
    if not connector or not connector.is_connected:
        return {}

    tables = connector.discover_tables()
    schema = {}
    for table in tables:
        try:
            cols = connector.describe_table(table)
            row_count = connector.get_table_row_count(table)
            schema[table] = {
                "columns": cols,
                "row_count": row_count,
            }
        except Exception as e:
            # 单表失败不阻塞其他表
            schema[table] = {"columns": [], "row_count": None, "error": str(e)}

    _log.info("Schema 发现完成: %d 张表", len(schema))
    return schema


def get_cached_schema() -> dict:
    """带 TTL 缓存的 Schema 获取（推荐使用）

    TTL 内直接返回缓存，过期后自动重新发现。
    线程安全，多请求并发时仅锁定读写区域。

    Returns:
        Schema 字典，格式同 discover_schema()
    """
    global _cache, _cache_time
    ttl = SCHEMA_CACHE_TTL_MINUTES * 60

    with _cache_lock:
        if _cache is not None and (time.time() - _cache_time) < ttl:
            return _cache

    # 缓存未命中或过期 → 重新发现
    schema = discover_schema()
    with _cache_lock:
        _cache = schema
        _cache_time = time.time()
    return schema


def refresh_schema() -> dict:
    """强制刷新 Schema 缓存并返回最新结果

    用于 UI 上的"刷新 Schema"按钮或数据库结构变更后。

    Returns:
        最新的 Schema 字典
    """
    global _cache, _cache_time
    schema = discover_schema()
    with _cache_lock:
        _cache = schema
        _cache_time = time.time()
    return schema
