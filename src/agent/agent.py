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

from src.agent.context import ToolContext
from src.agent.tools import ALL_TOOLS
from src.llm.deepseek_client import get_llm

# ── Agent 系统提示词 ──
AGENT_SYSTEM_PROMPT = """你是一个专业的数据分析 Agent。用户已上传数据文件，你需要通过调用工具来完成分析任务。

工作准则:
1. **先探索后分析**: 先用 describe_data 和 check_missing 了解数据结构，再做深入分析
2. **图表必配文字**: 每生成一张图表，都要用文字解释图表反映的信息
3. **中文输出**: 所有分析结论必须用中文表达
4. **洞察导向**: 不只是罗列数字，要给出业务洞察和建议
5. **工具路径**: 图表生成后会返回文件路径，请在最终回答中引用这些路径

分析报告格式:
- 先总结关键发现
- 再展示分析过程和数据
- 最后给出可操作的建议"""


def _create_agent(temperature: float = 0.1):
    """
    创建 ReAct Agent 实例

    使用 LangGraph 的 create_react_agent，自动实现:
      Thought → Action → Observation 循环

    Args:
        temperature: 模型创造性参数

    Returns:
        编译好的 Agent 图（可调用 .invoke()）
    """
    llm = get_llm(temperature=temperature, streaming=False)
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
    agent = _create_agent(temperature=temperature)
    result = agent.invoke({"messages": messages})

    # 解析 Agent 输出
    output_messages = result.get("messages", [])

    # 提取最终文本回复（最后一个 AI 消息）
    final_text = ""
    for msg in reversed(output_messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content
            break

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
