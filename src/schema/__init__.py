"""
MySQL 数据库 Schema 包
───────────────────────
提供数据库元数据的自动发现、缓存和可读化输出。

模块:
  discovery.py  - Schema 发现与缓存
  serializer.py - Schema 转 LLM 友好文本

设计要点:
  - Schema 有 TTL 缓存（默认 30 分钟），避免频繁查询 INFORMATION_SCHEMA
  - 序列化输出针对 LLM 上下文优化：列宽对齐、长文本截断、行数限制
  - 缓存线程安全（threading.Lock）
"""
from src.schema.discovery import discover_schema, get_cached_schema, refresh_schema
from src.schema.serializer import serialize_schema, serialize_table_schema
