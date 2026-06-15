"""
Langfuse 观测回调（基于 REST API）
─────────────────────────────────
自定义 LangChain BaseCallbackHandler，通过 Langfuse v2 REST API
直接创建 Trace / Span / Generation，不依赖不兼容的 langfuse.callback 模块。

用法:
  handler = LangfuseCallback(session_id="xxx", tags=["datamate"])
  agent.stream(messages, config={"callbacks": [handler]})
"""
from __future__ import annotations
import time
import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.observability.langfuse_client import get_langfuse
from src.logger import get_logger

_log = get_logger("observability.callback")

# 跨线程存储当前 trace 信息
_trace_storage: dict[int, dict[str, Any]] = {}
_lock = threading.Lock()


class LangfuseCallback(BaseCallbackHandler):
    """
    自定义 LangChain 回调处理器

    拦截 LLM 调用和 Tool 调用，通过 Langfuse REST API 上报。

    LLM 调用 → Generation span（记录 tokens、模型名、耗时）
    Tool 调用 → Tool span（记录输入输出）
    """

    def __init__(
        self,
        session_id: str = "",
        user_id: str = "default",
        tags: list[str] | None = None,
        trace_name: str = "datamate-request",
    ):
        self._session_id = session_id
        self._user_id = user_id
        self._tags = tags or []
        self._trace_name = trace_name

        self._client = get_langfuse()
        self._active_spans: dict[UUID, Any] = {}
        self._span_start_times: dict[UUID, float] = {}

        # 提前创建 Trace，确保 LLM 直接调用时也有 trace
        if self._client is not None:
            self._trace = self._client.trace(
                name=self._trace_name,
                session_id=self._session_id,
                user_id=self._user_id,
                tags=self._tags,
            )
            self._trace_id = self._trace.id
            self._root_span = None
            from src.observability.langfuse_client import ObservationContext
            ObservationContext.set_trace(self._trace_id)
        else:
            self._trace = None
            self._trace_id = None
            self._root_span = None

    # ── Agent / Chain 级别 ──

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._trace is None or self._client is None:
            return

        if parent_run_id is None:
            self._root_span = self._trace.span(
                name="agent",
                input=dict(inputs),
            )
            self._active_spans[run_id] = self._root_span
            self._span_start_times[run_id] = time.time()

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._active_spans.pop(run_id, None)
        if span:
            span.end(output=dict(outputs))
        self._span_start_times.pop(run_id, None)

    # ── LLM 调用级别 ──

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._trace is None or self._client is None:
            return

        model_name = serialized.get("kwargs", {}).get("model", "unknown")
        parent = self._active_spans.get(parent_run_id) if parent_run_id else self._root_span
        if parent is None:
            parent = self._trace

        generation = parent.generation(
            name="llm_call",
            model=model_name,
            input=prompts[0] if prompts else "",
        )
        self._active_spans[run_id] = generation
        self._span_start_times[run_id] = time.time()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        gen = self._active_spans.pop(run_id, None)
        if gen is None:
            return

        start = self._span_start_times.pop(run_id, time.time())
        duration_ms = int((time.time() - start) * 1000)

        # 提取 token 用量
        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for g in gen_list:
                    if hasattr(g, "generation_info") and g.generation_info:
                        usage = g.generation_info.get("usage_metadata") or {}
                        token_usage.update(usage)

        usage_obj = _map_token_usage(token_usage) if token_usage else None
        gen.end(
            output=str(response.generations[0][0].text if response.generations else ""),
            usage=usage_obj,
        )

    # ── Tool 调用级别 ──

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._trace is None or self._client is None:
            return

        tool_name = serialized.get("name", "unknown")
        parent = self._active_spans.get(parent_run_id) if parent_run_id else self._root_span
        if parent is None:
            parent = self._trace

        span = parent.span(
            name=tool_name,
            input=input_str[:500] if input_str else "",
        )
        self._active_spans[run_id] = span
        self._span_start_times[run_id] = time.time()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._active_spans.pop(run_id, None)
        if span:
            span.end(output=str(output)[:2000] if output else "")

        self._span_start_times.pop(run_id, None)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._active_spans.pop(run_id, None)
        if span:
            span.end(level="ERROR", status_message=str(error))
        self._span_start_times.pop(run_id, None)

    # ── 错误处理 ──

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._active_spans.pop(run_id, None)
        if span:
            span.end(level="ERROR", status_message=str(error)[:500])
        self._span_start_times.pop(run_id, None)

    # ── 获取 trace 信息 ──

    @property
    def trace_id(self) -> str | None:
        return self._trace_id


def create_handler(
    session_id: str = "",
    user_id: str = "default",
    tags: list[str] | None = None,
    trace_name: str = "datamate-request",
) -> LangfuseCallback | None:
    """
    创建 Langfuse 回调处理器

    Args:
        session_id: 会话 ID
        user_id:    用户标识
        tags:       标签列表
        trace_name: Trace 名称

    Returns:
        LangfuseCallback 实例；若 Langfuse 不可用则返回 None
    """
    if get_langfuse() is None:
        return None

    return LangfuseCallback(
        session_id=session_id,
        user_id=user_id,
        tags=tags or ["datamate"],
        trace_name=trace_name,
    )


def build_metadata(
    session_id: str = "",
    user_id: str = "default",
    trace_name: str = "",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 LangChain config metadata（兼容层，实际数据由 LangfuseCallback 管理）"""
    return {}


def _map_token_usage(raw: dict[str, Any]) -> dict[str, Any] | None:
    """
    将 DeepSeek/OpenAI 格式的 token 统计映射为 Langfuse v2 期望的 camelCase

    DeepSeek 返回: {prompt_tokens, completion_tokens, total_tokens}
    Langfuse 期望: {promptTokens, completionTokens, totalTokens} 或 {input, output, total, unit}

    优先使用驼峰格式，若不匹配则使用 input/output 格式。
    """
    mapped: dict[str, Any] = {}

    if "prompt_tokens" in raw:
        mapped["promptTokens"] = raw["prompt_tokens"]
    elif "input" in raw:
        mapped["input"] = raw["input"]

    if "completion_tokens" in raw:
        mapped["completionTokens"] = raw["completion_tokens"]
    elif "output" in raw:
        mapped["output"] = raw["output"]

    if "total_tokens" in raw:
        mapped["totalTokens"] = raw["total_tokens"]
    elif "total" in raw:
        mapped["total"] = raw["total"]

    if not mapped:
        return None

    if "totalTokens" not in mapped and "total" not in mapped:
        pt = mapped.get("promptTokens", 0) or mapped.get("input", 0) or 0
        ct = mapped.get("completionTokens", 0) or mapped.get("output", 0) or 0
        mapped["totalTokens"] = pt + ct

    return mapped
