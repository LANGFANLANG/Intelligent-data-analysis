"""
Langfuse 客户端模块
───────────────────
封装 Langfuse SDK，提供单例客户端和观测上下文管理。

Langfuse 是一个开源 LLM 观测平台，支持:
  - 全链路 Trace（LLM 调用 + Tool 执行）
  - Token 消耗统计与成本计算
  - Prompt 版本管理
  - 人工反馈与自动评测

用法:
  from src.observability import get_langfuse, ObservationContext

  client = get_langfuse()
  ObservationContext.init(session_id="xxx", user_id="user1")

  with ObservationContext.trace("my_operation"):
      # LLM 调用和 Tool 调用自动被 CallbackHandler 拦截上报
      ...
"""
import os
import contextvars
from dataclasses import dataclass, field
from typing import Any

from src.logger import get_logger

_log = get_logger("observability")

# ── 全局单例 ──
_langfuse = None


def get_langfuse():
    """
    获取 Langfuse 客户端单例（懒加载）

    Returns:
        Langfuse 客户端实例；若未配置则返回 None
    """
    global _langfuse
    if _langfuse is None:
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

        if not secret_key or not public_key or "xxx" in secret_key:
            _log.warning("Langfuse 未配置（缺少 LANGFUSE_SECRET_KEY / PUBLIC_KEY），观测功能已禁用")
            return None

        try:
            from langfuse import Langfuse
            _langfuse = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=host,
            )
            _log.info("Langfuse 客户端初始化成功，host=%s", host)
        except ImportError:
            _log.warning("langfuse 包未安装，观测功能已禁用")
            return None
        except Exception as e:
            _log.warning("Langfuse 客户端初始化失败: %s，观测功能已禁用", e)
            return None

    return _langfuse


# ── 观测上下文（contextvars，跨函数隐式传播）──

# 当前 trace ID（由 Langfuse 自动生成）
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("obs_trace_id", default=None)
# 当前 trace 的 Langfuse URL（用于 UI 跳转）
_trace_url_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("obs_trace_url", default=None)
# 当前会话 ID
_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("obs_session_id", default=None)


@dataclass
class TraceInfo:
    """单次请求的观测信息"""
    trace_id: str | None = None
    trace_url: str | None = None
    session_id: str | None = None
    user_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


class ObservationContext:
    """
    观测上下文管理器

    替代原有的 TraceContext，提供兼容 API，
    底层由 Langfuse CallbackHandler 自动管理 Trace 生命周期。
    """

    @classmethod
    def init(cls, session_id: str = "", user_id: str = "default", **metadata) -> str:
        """
        开始新的观测上下文

        Args:
            session_id: 会话 ID（关联 Langfuse Session）
            user_id:    用户标识
            **metadata: 附加元数据（tags, version 等）

        Returns:
            session_id
        """
        _session_id_var.set(session_id)
        cls.add_meta(session_id=session_id, user_id=user_id, **metadata)
        _log.debug("观测上下文初始化: session=%s", session_id[:8] if session_id else "None")
        return session_id

    @classmethod
    def set_trace(cls, trace_id: str, trace_url: str = ""):
        """记录当前 trace 信息（由 handler 回调设置）"""
        _trace_id_var.set(trace_id)
        _trace_url_var.set(trace_url)

    @classmethod
    def get_trace_id(cls) -> str | None:
        """获取当前 trace ID"""
        return _trace_id_var.get()

    @classmethod
    def get_trace_url(cls) -> str | None:
        """获取当前 trace 在 Langfuse UI 中的链接"""
        return _trace_url_var.get()

    @classmethod
    def get_session_id(cls) -> str | None:
        """获取当前会话 ID"""
        return _session_id_var.get()

    @classmethod
    def add_meta(cls, **kwargs):
        """暂存元数据（后续 flush 到 trace）"""
        pass

    @classmethod
    def add_token(cls, count: int = 1):
        """Token 计数由 Langfuse 自动统计，此方法保留为兼容空操作"""
        pass

    @classmethod
    def get_token_count(cls) -> int:
        """返回 0，实际 token 统计见 Langfuse UI"""
        return 0

    @classmethod
    def is_enabled(cls) -> bool:
        """检查 Langfuse 是否可用"""
        return get_langfuse() is not None


def score_trace(
    trace_id: str,
    name: str = "user_feedback",
    value: float = 1.0,
    comment: str = "",
) -> bool:
    """
    对 Langfuse Trace 进行打分（人工反馈 / 自动评测）

    Args:
        trace_id: 要打分的 Trace ID
        name:     评分名称（如 "user_feedback", "quality", "accuracy"）
        value:    分数值（0.0 ~ 1.0，1.0 为最佳）
        comment:  评分备注

    Returns:
        是否成功上报
    """
    client = get_langfuse()
    if client is None:
        return False

    try:
        client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )
        _log.debug("Trace 评分已上报: trace=%s, name=%s, value=%.2f", trace_id[:8], name, value)
        return True
    except Exception as e:
        _log.debug("Trace 评分上报失败: %s", e)
        return False
