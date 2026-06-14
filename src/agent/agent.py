"""
ReAct Agent 核心模块
───────────────────
基于 LangGraph 的 create_react_agent 构建数据分析 Agent，
自动在思考-行动-观察循环中调用工具完成分析任务。

核心流程:
  1. 用户上传数据 + 输入问题
  2. Agent 思考 → 选择工具 → 执行 → 观察结果 → 继续思考...
  3. 最终输出: Markdown 文本报告 + 可视化图表路径

依赖:
  - src.agent.context: 工具上下文（持有 DataFrame）
  - src.agent.tools:   已注册的 LangChain Tool 列表
  - src.llm:            DeepSeek 模型客户端
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.agent.context import ToolContext
from src.agent.tools import ALL_TOOLS
from src.llm.deepseek_client import get_llm
from src.config import DEEPSEEK_MODEL
from src.tracing import trace_span, TraceContext

# ── Agent 系统提示词 ──
AGENT_SYSTEM_PROMPT = """你是一个专业的数据分析 Agent。用户已上传数据文件，你需要通过调用工具来完成分析任务。

工作准则:
1. **先探索后分析**: 先用 describe_data 和 check_missing 了解数据结构
2. **按需可视化**: 只在用户明确要求"画图/可视化/趋势/对比图/分布"等时才调用图表工具（line_chart/bar_chart/scatter_plot/pie_chart/heatmap/histogram）
3. **简单问题直达**: 计数、均值、排序、分组统计等简单问题直接用 describe_data/groupby_agg/value_counts 等文字工具回答，不画图
4. **中文输出**: 所有分析结论必须用中文表达
5. **洞察导向**: 不只罗列数字，要给出业务洞察和建议
6. **图表配文字**: 如果生成了图表，必须用文字解释图表反映的信息

分析报告格式:
- 先总结关键发现
- 再展示分析过程和数据
- 最后给出可操作的建议（如有）"""


def _create_agent(temperature: float = 0.1, streaming: bool = True):
    """
    创建 ReAct Agent 实例

    使用 LangGraph 的 create_react_agent，自动实现:
      Thought → Action → Observation 循环

    Args:
        temperature: 模型创造性参数
        streaming:   是否启用 LLM 流式输出

    Returns:
        编译好的 Agent 图（可调用 .invoke() 或 .stream()）
    """
    llm = get_llm(temperature=temperature, streaming=streaming)
    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=AGENT_SYSTEM_PROMPT,
    )


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
    if not ToolContext.has_data():
        return {
            "text": "请先上传数据文件后再开始分析。",
            "images": [],
        }

    # 构建消息列表：系统提示 + 历史对话 + 当前问题
    messages = []

    # 注入数据上下文信息，帮助 Agent 更快了解数据结构
    df = ToolContext.get()
    data_info = f"[当前数据: {ToolContext.get_name()}, {df.shape[0]}行x{df.shape[1]}列, 列名: {', '.join(df.columns)}]"

    if history:
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=f"{data_info}\n\n用户问题: {prompt}"))

    # 创建并执行 Agent
    agent = _create_agent(temperature=temperature, streaming=False)

    # 兜底机制：限制最多40步(思考+行动)，120秒超时
    # 复杂分析任务（多图表+多查询）可能需要更多步骤
    config = {"recursion_limit": 40}
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.invoke, {"messages": messages}, config)
            result = future.result(timeout=120)
    except FuturesTimeoutError:
        return {
            "text": "分析超时(120秒)，请尝试简化问题或检查数据质量。",
            "images": [],
        }
    except Exception as e:
        error_str = str(e)
        # recursion_limit 耗尽时 Agent 会抛特定异常
        if "recursion" in error_str.lower() or "limit" in error_str.lower():
            return {
                "text": "分析步骤过多(超过40步)，建议将问题拆分为多个小问题依次提问。",
                "images": [],
            }
        raise  # 其他异常继续向上抛出，由 UI 层兜底

    # 解析 Agent 输出
    output_messages = result.get("messages", [])

    # 提取最终文本回复（最后一个 AI 消息）
    final_text = ""
    for msg in reversed(output_messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content
            break

    # 检测 LLM 是否提示步骤不足，追加用户指引
    need_more_hints = {"need more steps", "need more tool", "need additional step",
                       "need more information", "more steps to process",
                       "unable to complete", "cannot complete",
                       "需要更多步骤", "步骤不足", "无法完成"}
    final_lower = final_text.lower() if final_text else ""
    if any(hint in final_lower for hint in need_more_hints):
        final_text += ("\n\n> 分析未完全结束，建议将问题拆分为更小的子问题依次提问，"
                       "或直接指明具体需要什么样的分析。")

    # 提取图表文件路径（ToolMessage 中返回的 .png 路径）
    image_paths = []
    for msg in output_messages:
        content = str(msg.content) if hasattr(msg, "content") else str(msg)
        # 检查是否为图片路径
        if ".png" in content and ("outputs" in content or "\\" in content or "/" in content):
            for line in content.split("\n"):
                line = line.strip()
                if line.endswith(".png"):
                    image_paths.append(line)

    # 去重
    image_paths = list(dict.fromkeys(image_paths))

    return {
        "text": final_text if final_text else "分析完成，但未能生成文本报告。",
        "images": image_paths,
    }


def run_agent_stream(prompt: str, history: list[dict] | None = None, temperature: float = 0.1):
    """
    流式执行 Agent 分析任务，逐步 yield 事件供 UI 渐进渲染

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

    if not ToolContext.has_data():
        yield {"type": "error", "content": "请先上传数据文件后再开始分析。"}
        return

    # 构建消息列表
    messages = []
    df = ToolContext.get()
    data_info = f"[当前数据: {ToolContext.get_name()}, {df.shape[0]}行x{df.shape[1]}列, 列名: {', '.join(df.columns)}]"

    if history:
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=f"{data_info}\n\n用户问题: {prompt}"))

    # 创建流式 Agent
    agent = _create_agent(temperature=temperature, streaming=True)
    config = {"recursion_limit": 40}

    full_text = ""
    images = []
    tool_calls_seen = set()
    llm_call_index = 0

    with trace_span("run_agent_stream"):
        TraceContext.add_meta(model=DEEPSEEK_MODEL)
        try:
            for msg, _meta in agent.stream(
                {"messages": messages}, config, stream_mode="messages"
            ):
                if isinstance(msg, AIMessageChunk):
                    # 工具调用检测：从 tool_calls 增量中提取新工具名
                    if msg.tool_calls:
                        new_tools = []
                        for tc in msg.tool_calls:
                            name = tc.get("name")
                            if name and name not in tool_calls_seen:
                                tool_calls_seen.add(name)
                                new_tools.append(name)
                        if new_tools:
                            llm_call_index += 1
                            TraceContext.add_meta(
                                **{f"llm_call_{llm_call_index}_tools": new_tools}
                            )
                            for t in new_tools:
                                yield {"type": "tool_start", "tool": t}

                    # 文本 token 输出
                    if msg.content and isinstance(msg.content, str):
                        full_text += msg.content
                        TraceContext.add_token()
                        yield {"type": "token", "content": msg.content}

                elif isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", "unknown")
                    content = str(msg.content) if msg.content else ""
                    tool_imgs = []

                    with trace_span(tool_name):
                        # 记录工具执行状态
                        if content.startswith("错误"):
                            TraceContext.add_meta(
                                result_status="error",
                                error_msg=content[:200],
                            )
                        elif ".png" in content:
                            # 工具返回了图片路径
                            png_count = content.count(".png")
                            TraceContext.add_meta(
                                result_status="success",
                                has_image=True,
                                image_count=png_count,
                            )
                        else:
                            TraceContext.add_meta(
                                result_status="success",
                                result_len=len(content),
                            )

                        # 从工具输出中提取 .png 图片路径
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
                    # 兜底：如果 LLM 以非流式返回完整消息，取内容
                    if not full_text and msg.content:
                        content = str(msg.content)
                        full_text = content
                        TraceContext.add_token()
                        yield {"type": "token", "content": content}

        except Exception as e:
            error_str = str(e)
            if "recursion" in error_str.lower() or "limit" in error_str.lower():
                yield {"type": "error", "content": "分析步骤过多(超过40步)，建议将问题拆分为多个小问题依次提问。"}
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

        TraceContext.add_meta(
            total_tokens=TraceContext.get_token_count(),
            llm_calls=llm_call_index,
        )

        yield {"type": "done", "full_text": full_text, "images": images}


CHAT_SYSTEM_PROMPT = """你是一个友好的 AI 助手 DataMate，擅长数据分析相关的技术问答和日常对话。

你可以:
- 回答数据分析、统计学、机器学习相关的概念问题
- 解释图表、数据指标的含义
- 提供数据分析的方法论建议
- 进行日常闲聊、问候

当用户需要分析具体数据时，引导他们先在左侧上传数据文件。

请用中文回答，语气友好、专业。"""


def run_chat_stream(prompt: str, history: list[dict] | None = None, temperature: float = 0.7):
    """
    流式执行聊天 Agent（无工具），用于非数据分析的日常对话

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

    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]

    if history:
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=prompt))

    llm = get_llm(temperature=temperature, streaming=True)
    full_text = ""

    with trace_span("chat_llm_call"):
        TraceContext.add_meta(model=DEEPSEEK_MODEL)
        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    full_text += chunk.content
                    TraceContext.add_token()
                    yield {"type": "token", "content": chunk.content}
        except Exception as e:
            yield {"type": "error", "content": f"对话异常: {e}"}
            return

        TraceContext.add_meta(total_tokens=TraceContext.get_token_count())
        yield {"type": "done", "full_text": full_text, "images": []}
