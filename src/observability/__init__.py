"""
观测模块 (Observability)
────────────────────────
基于 Langfuse REST API 的自定义 LLM 观测层。

提供:
  - get_langfuse():         获取 Langfuse 客户端单例
  - create_handler():       创建自定义 LangChain CallbackHandler
  - get_prompt():           从 Langfuse 拉取托管 Prompt（支持版本管理 + 本地 fallback）
  - score_trace():          对 Trace 进行打分（人工反馈）
  - ObservationContext:     观测上下文管理（session_id, trace_id）
"""

from src.observability.langfuse_client import (
    get_langfuse,
    ObservationContext,
    score_trace,
)
from src.observability.handler import create_handler, LangfuseCallback
from src.observability.prompt_manager import get_prompt, PromptManager, DEFAULT_PROMPTS

__all__ = [
    "get_langfuse",
    "ObservationContext",
    "score_trace",
    "create_handler",
    "LangfuseCallback",
    "get_prompt",
    "PromptManager",
    "DEFAULT_PROMPTS",
]
