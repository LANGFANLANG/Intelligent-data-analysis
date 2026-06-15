"""
意图识别路由模块
───────────────
在用户消息进入 Agent 之前做意图分类，将不同意图分发到不同的处理路径:

  - "analysis":  数据已上传 → ReAct Agent (17 个工具)
                  数据未上传 → 提示先上传
  - "chat":      纯聊天 Agent (无工具，节省 token)

设计原则:
  - 分类用轻量 LLM 调用 (~0.3s)，远快于完整分析
  - 两个路径的事件格式完全一致，UI 层无感知
  - 数据就绪检查仅在 analysis 分支执行
  - 分类提示词托管在 Langfuse Prompt Management，支持版本管理
"""
from src.agent.context import ToolContext
from src.agent.agent import run_agent_stream, run_chat_stream
from src.llm.deepseek_client import get_llm
from src.llm.rate_limiter import rate_limiter
from src.observability import ObservationContext, get_prompt, create_handler
from langchain_core.messages import HumanMessage, SystemMessage


def classify_intent(prompt: str) -> str:
    """
    用轻量 LLM 调用判断用户意图

    提示词从 Langfuse Prompt Management 获取，
    不可用时 fallback 到本地默认值。

    Args:
        prompt: 用户输入文本

    Returns:
        "analysis" 或 "chat"
    """
    handler = create_handler(
        session_id=ObservationContext.get_session_id() or "",
        tags=["classify-intent"],
        trace_name="classify_intent",
    )
    callbacks = [handler] if handler else []
    llm = get_llm(temperature=0, callbacks=callbacks)
    classify_prompt = get_prompt("datamate-router-system")
    messages = [
        SystemMessage(content=classify_prompt),
        HumanMessage(content=f"用户输入: {prompt}"),
    ]
    try:
        response = llm.invoke(messages, config={"callbacks": callbacks})
        result = str(response.content).strip().upper()
        if "ANALYSIS" in result:
            return "analysis"
        return "chat"
    except Exception:
        return "chat"


def route_stream(prompt: str, history: list[dict] | None = None, temperature: float = 0.1):
    """
    统一流式入口: 先分类意图，再分发到对应处理路径

    Args:
        prompt:      用户输入
        history:     历史对话记录
        temperature: 模型创造性参数

    Yields:
        与 run_agent_stream / run_chat_stream 相同的事件格式:
          {"type": "token",      "content": "字"}
          {"type": "tool_start", "tool": "name"}
          {"type": "tool_end",   "tool": "name", "images": [...]}
          {"type": "error",      "content": "..."}
          {"type": "done",       "full_text": "...", "images": [...]}
    """
    if not rate_limiter.acquire():
        yield {"type": "error", "content": "请求过于频繁，请稍后重试（每分钟限制 30 次）"}
        return

    intent = classify_intent(prompt)
    has_data = ToolContext.has_data()
    is_db = ToolContext.is_database_mode()

    if intent == "analysis":
        if not has_data and not is_db:
            yield {"type": "error", "content": "请先在左侧上传数据文件或配置数据库连接后再开始分析。"}
            return
        yield from run_agent_stream(prompt, history, temperature)
    else:
        yield from run_chat_stream(prompt, history, temperature)
