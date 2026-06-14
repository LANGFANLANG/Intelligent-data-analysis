"""
全链路追踪模块
─────────────
基于 contextvars 的轻量级分布式追踪，为每次用户请求构建完整的 span 调用树。

Span 层级示例:
  user_request
  ├── classify_intent
  ├── route
  │   └── run_agent_stream
  │       ├── llm_call
  │       ├── tool_describe_data
  │       ├── llm_call
  │       ├── tool_line_chart
  │       └── llm_call
  └── chat_llm_call

用法:
  from src.tracing import TraceContext, trace_span, TraceManager

  TraceContext.init()                    # 入口: 开启新链路

  with trace_span("classify_intent"):
      result = classify(prompt)
      TraceContext.add_meta(intent=result)

  with trace_span("run_agent_stream"):
      ...

  TraceManager.save(session_id)          # 出口: 批量写入 DB
"""
import uuid
import time
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── 上下文变量（隐式传播，跨函数调用无需显式传参）──
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_current_span_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_span_id", default=None)
_token_count: contextvars.ContextVar[int] = contextvars.ContextVar("token_count", default=0)


@dataclass
class TraceSpan:
    """单个调用跨度"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    name: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_ms: int | None = None
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceContext:
    """基于 contextvars 的链路上下文"""

    @classmethod
    def init(cls) -> str:
        """开启新链路，返回 trace_id"""
        trace_id = str(uuid.uuid4())
        _trace_id_var.set(trace_id)
        _current_span_var.set(None)
        _token_count.set(0)
        TraceManager._spans.clear()
        return trace_id

    @classmethod
    def get_trace_id(cls) -> str | None:
        return _trace_id_var.get()

    @classmethod
    def get_current_span_id(cls) -> str | None:
        return _current_span_var.get()

    @classmethod
    def add_meta(cls, **kwargs):
        """向当前 span 追加 metadata"""
        span_id = _current_span_var.get()
        if span_id and span_id in TraceManager._spans:
            TraceManager._spans[span_id].metadata.update(kwargs)

    @classmethod
    def add_token(cls, count: int = 1):
        _token_count.set(_token_count.get() + count)

    @classmethod
    def get_token_count(cls) -> int:
        return _token_count.get()

    @classmethod
    def finish(cls) -> list[TraceSpan]:
        """结束链路，返回所有 span"""
        return list(TraceManager._spans.values())


class TraceManager:
    """Span 生命周期管理器"""

    _spans: dict[str, TraceSpan] = {}

    @classmethod
    def start_span(cls, name: str, parent_id: str | None = None, **metadata) -> TraceSpan:
        span = TraceSpan(
            name=name,
            parent_id=parent_id or _current_span_var.get(),
            metadata=dict(metadata),
        )
        cls._spans[span.span_id] = span
        _current_span_var.set(span.span_id)
        return span

    @classmethod
    def end_span(cls, span: TraceSpan, status: str = "success", **extra_meta):
        span.end_time = datetime.now(timezone.utc)
        span.duration_ms = int((span.end_time - span.start_time).total_seconds() * 1000)
        span.status = status
        span.metadata.update(extra_meta)
        # 恢复父 span 为当前
        _current_span_var.set(span.parent_id)

    @classmethod
    def save(cls, session_id, spans: list[TraceSpan] | None = None):
        """批量写入 DB"""
        if spans is None:
            spans = list(cls._spans.values())
        if not spans:
            return

        from src.database.db import SessionLocal
        from src.database.models import Trace

        db = SessionLocal()
        try:
            trace_id = _trace_id_var.get()
            rows = []
            for s in spans:
                rows.append(Trace(
                    id=s.span_id,
                    session_id=session_id,
                    trace_id=trace_id or "",
                    parent_id=s.parent_id,
                    name=s.name,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    duration_ms=s.duration_ms,
                    status=s.status,
                    metadata_=s.metadata,
                ))
            db.add_all(rows)
            db.commit()
        except Exception as e:
            db.rollback()
            from src.logger import get_logger
            get_logger("trace").error("保存链路失败: %s", e, exc_info=True)
        finally:
            db.close()


@contextmanager
def trace_span(name: str, **metadata):
    """
    上下文管理器：自动记录 span 的开始/结束/异常

    Usage:
        with trace_span("classify_intent"):
            result = classify(prompt)
    """
    span = TraceManager.start_span(name, **metadata)
    try:
        yield span
        TraceManager.end_span(span, "success")
    except Exception as e:
        TraceManager.end_span(span, "error", error=str(e))
        raise
