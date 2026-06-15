"""
ReAct Agent 核心模块
───────────────────
基于 LangGraph 的 create_react_agent 构建数据分析 Agent，
自动在思考-行动-观察循环中调用工具完成分析任务。

核心流程:
  1. 用户上传数据 + 输入问题
  2. Agent 思考 → 选择工具 → 执行 → 观察结果 → 继续思考...
  3. 最终输出: Markdown 文本报告 + 可视化图表路径

观测集成:
  - 通过 Langfuse CallbackHandler 自动拦截 LLM 调用和 Tool 执行
  - Prompt 通过 Langfuse Prompt Management 托管，支持版本化

依赖:
  - src.agent.context: 工具上下文（持有 DataFrame）
  - src.agent.tools:   已注册的 LangChain Tool 列表
  - src.llm:            DeepSeek 模型客户端
  - src.observability:  Langfuse 观测层
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.agent.context import ToolContext
from src.agent.tools import ALL_TOOLS, DB_TOOLS
from src.llm.deepseek_client import get_llm
from src.config import DEEPSEEK_MODEL, AGENT_MAX_STEPS, AGENT_TIMEOUT_SECONDS
from src.observability import create_handler, ObservationContext, get_prompt

# ── Agent 系统提示词 ──
# 提示词已迁移到 Langfuse Prompt Management 和 src/observability/prompt_manager.py
# get_prompt("datamate-agent-system") → 从 Langfuse 拉取，不可用时 fallback 到本地默认值


def _create_agent(temperature: float = 0.1, streaming: bool = True, is_db_mode: bool = False):
    """
    创建 ReAct Agent 实例

    使用 LangGraph 的 create_react_agent，自动实现:
      Thought → Action → Observation 循环

    系统提示词从 Langfuse Prompt Management 获取（支持版本管理），
    不可用时自动 fallback 到本地默认值。

    Args:
        temperature: 模型创造性参数
        streaming:   是否启用 LLM 流式输出
        is_db_mode:  是否使用数据库分析模式（不同的工具集和提示词）

    Returns:
        (agent, handler) 元组:
          - agent:   编译好的 Agent 图（可调用 .invoke() 或 .stream()）
          - handler: Langfuse CallbackHandler（需传入 agent.stream/invoke 的 config.callbacks）
    """
    handler = create_handler(
        session_id=ObservationContext.get_session_id() or "",
        tags=[DEEPSEEK_MODEL, "agent"],
        trace_name="agent_analysis",
    )
    callbacks = [handler] if handler else []

    llm = get_llm(temperature=temperature, streaming=streaming, callbacks=callbacks)
    tools = DB_TOOLS if is_db_mode else ALL_TOOLS

    prompt_name = "datamate-db-agent-system" if is_db_mode else "datamate-agent-system"
    prompt = get_prompt(prompt_name)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )
    return agent, callbacks


def run_agent(prompt: str, history: list[dict] | None = None, temperature: float = 0.1) -> dict:
    """
    执行 Agent 分析任务

    Args:
        prompt:      用户的分析需求（自然语言）
        history:     历史对话记录，格式: [{"role": "user"|"assistant", "content": "..."}, ...]
        temperature: 模型创造性参数

    Returns:
        {
            "text":   "Markdown 格式的分析报告文本",
            "images": ["outputs/xxx.png", "outputs/yyy.png", ...],  # 图表文件路径列表
        }
    """
    is_db = ToolContext.is_database_mode()

    if not is_db and not ToolContext.has_data():
        return {
            "text": "请先上传数据文件或配置数据库连接后再开始分析。",
            "images": [],
        }

    messages = []

    if is_db:
        from src.schema import get_cached_schema, serialize_schema
        schema = get_cached_schema()
        schema_text = serialize_schema(schema)
        data_info = f"[数据库已连接, 可用表和结构如下]\n\n{schema_text}"
    else:
        df = ToolContext.get()
        data_info = f"[当前数据: {ToolContext.get_name()}, {df.shape[0]}行x{df.shape[1]}列, 列名: {', '.join(df.columns)}]"

    if history:
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=f"{data_info}\n\n用户问题: {prompt}"))

    agent, callbacks = _create_agent(temperature=temperature, streaming=False, is_db_mode=is_db)

    config = {"recursion_limit": AGENT_MAX_STEPS, "callbacks": callbacks}
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.invoke, {"messages": messages}, config)
            result = future.result(timeout=AGENT_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        return {
            "text": f"分析超时({AGENT_TIMEOUT_SECONDS}秒)，请尝试简化问题或检查数据质量。",
            "images": [],
        }
    except Exception as e:
        error_str = str(e)
        if "recursion" in error_str.lower() or "limit" in error_str.lower():
            return {
                "text": f"分析步骤过多(超过{AGENT_MAX_STEPS}步)，建议将问题拆分为多个小问题依次提问。",
                "images": [],
            }
        raise

    output_messages = result.get("messages", [])

    final_text = ""
    for msg in reversed(output_messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content
            break

    need_more_hints = {"need more steps", "need more tool", "need additional step",
                       "need more information", "more steps to process",
                       "unable to complete", "cannot complete",
                       "需要更多步骤", "步骤不足", "无法完成"}
    final_lower = final_text.lower() if final_text else ""
    if any(hint in final_lower for hint in need_more_hints):
        final_text += ("\n\n> 分析未完全结束，建议将问题拆分为更小的子问题依次提问，"
                       "或直接指明具体需要什么样的分析。")

    image_paths = []
    for msg in output_messages:
        content = str(msg.content) if hasattr(msg, "content") else str(msg)
        if ".png" in content and ("outputs" in content or "\\" in content or "/" in content):
            for line in content.split("\n"):
                line = line.strip()
                if line.endswith(".png"):
                    image_paths.append(line)

    image_paths = list(dict.fromkeys(image_paths))

    return {
        "text": final_text if final_text else "分析完成，但未能生成文本报告。",
        "images": image_paths,
    }


def run_agent_stream(prompt: str, history: list[dict] | None = None, temperature: float = 0.1):
    """
    流式执行 Agent 分析任务，逐步 yield 事件供 UI 渐进渲染

    LLM 调用和 Tool 执行由 Langfuse CallbackHandler 自动拦截上报，
    无需手动管理 Trace/Span 生命周期。

    Args:
        prompt:      用户的分析需求（自然语言）
        history:     历史对话记录
        temperature: 模型创造性参数

    Yields:
        {"type": "token",      "content": "字"}
        {"type": "tool_start", "tool": "tool_name"}
        {"type": "tool_end",   "tool": "name", "images": ["o/xxx.png"]}
        {"type": "error",      "content": "错误信息"}
        {"type": "done",       "full_text": "...", "images": [...]}
    """
    from langchain_core.messages import AIMessageChunk, AIMessage, ToolMessage, HumanMessage

    is_db = ToolContext.is_database_mode()
    if not is_db and not ToolContext.has_data():
        yield {"type": "error", "content": "请先上传数据文件或配置数据库连接后再开始分析。"}
        return

    messages = []

    if is_db:
        from src.schema import get_cached_schema, serialize_schema
        schema = get_cached_schema()
        schema_text = serialize_schema(schema)
        data_info = f"[数据库已连接, 可用表和结构如下]\n\n{schema_text}"
    else:
        df = ToolContext.get()
        data_info = f"[当前数据: {ToolContext.get_name()}, {df.shape[0]}行x{df.shape[1]}列, 列名: {', '.join(df.columns)}]"

    if history:
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=f"{data_info}\n\n用户问题: {prompt}"))

    agent, callbacks = _create_agent(temperature=temperature, streaming=True, is_db_mode=is_db)

    config = {"recursion_limit": AGENT_MAX_STEPS, "callbacks": callbacks}

    full_text = ""
    images = []
    tool_calls_seen = set()

    try:
        for msg, _meta in agent.stream(
            {"messages": messages}, config, stream_mode="messages"
        ):
            if isinstance(msg, AIMessageChunk):
                if msg.tool_calls:
                    new_tools = []
                    for tc in msg.tool_calls:
                        name = tc.get("name")
                        if name and name not in tool_calls_seen:
                            tool_calls_seen.add(name)
                            new_tools.append(name)
                    if new_tools:
                        for t in new_tools:
                            yield {"type": "tool_start", "tool": t}

                if msg.content and isinstance(msg.content, str):
                    full_text += msg.content
                    yield {"type": "token", "content": msg.content}

            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                content = str(msg.content) if msg.content else ""
                tool_imgs = []

                if ".png" in content:
                    for line in content.split("\n"):
                        line = line.strip()
                        if line.endswith(".png") and (
                            "outputs" in line or "/" in line or "\\" in line
                        ):
                            tool_imgs.append(line)

                images.extend(tool_imgs)
                yield {
                    "type": "tool_end",
                    "tool": tool_name,
                    "content": content,
                    "images": tool_imgs,
                }

            elif isinstance(msg, AIMessage):
                if not full_text and msg.content:
                    content = str(msg.content)
                    full_text = content
                    yield {"type": "token", "content": content}

    except Exception as e:
        error_str = str(e)
        if "recursion" in error_str.lower() or "limit" in error_str.lower():
            yield {"type": "error", "content": f"分析步骤过多(超过{AGENT_MAX_STEPS}步)，建议将问题拆分为多个小问题依次提问。"}
        else:
            yield {"type": "error", "content": f"分析异常: {error_str}"}
        return

    # 检测 LLM 是否提示步骤不足
    need_more_hints = {
        "need more steps", "need more tool", "need additional step",
        "need more information", "more steps to process",
        "unable to complete", "cannot complete",
        "需要更多步骤", "步骤不足", "无法完成",
    }
    final_lower = full_text.lower()
    if any(hint in final_lower for hint in need_more_hints):
        full_text += (
            "\n\n> 分析未完全结束，建议将问题拆分为更小的子问题依次提问，"
            "或直接指明具体需要什么样的分析。"
        )

    # 去重图片路径
    images = list(dict.fromkeys(images))

    yield {"type": "done", "full_text": full_text, "images": images}




def run_chat_stream(prompt: str, history: list[dict] | None = None, temperature: float = 0.7):
    """
    流式执行聊天 Agent（无工具），用于非数据分析的日常对话

    系统提示词从 Langfuse Prompt Management 获取，不可用时 fallback 到本地默认值。

    Args:
        prompt:      用户输入
        history:     历史对话记录
        temperature: 模型创造性参数（聊天时默认较高，更有趣）

    Yields:
        {"type": "token", "content": "字"}
        {"type": "error", "content": "..."}
        {"type": "done",  "full_text": "...", "images": []}
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from src.llm.deepseek_client import get_llm

    chat_prompt = get_prompt("datamate-chat-system")
    messages = [SystemMessage(content=chat_prompt)]

    if history:
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=prompt))

    handler = create_handler(
        session_id=ObservationContext.get_session_id() or "",
        tags=[DEEPSEEK_MODEL, "chat"],
        trace_name="chat",
    )
    callbacks = [handler] if handler else []
    llm = get_llm(temperature=temperature, streaming=True, callbacks=callbacks)
    full_text = ""

    try:
        for chunk in llm.stream(messages, config={"callbacks": callbacks}):
            if chunk.content:
                full_text += chunk.content
                yield {"type": "token", "content": chunk.content}
    except Exception as e:
        yield {"type": "error", "content": f"对话异常: {e}"}
        return

    yield {"type": "done", "full_text": full_text, "images": []}
