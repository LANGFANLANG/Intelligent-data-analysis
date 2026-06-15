"""
追踪模块（已迁移到 observability）
─────────────────────────────────
此模块已废弃，所有追踪功能已迁移到 src.observability（基于 Langfuse）。

保留此文件作为兼容层，重导出观测模块的核心组件。
建议新代码直接从 src.observability 导入。
"""
from src.observability import (
    ObservationContext as TraceContext,
    get_langfuse,
    create_handler,
)
from src.observability.langfuse_client import ObservationContext as _ObsCtx

__all__ = ["TraceContext", "get_langfuse", "create_handler"]

