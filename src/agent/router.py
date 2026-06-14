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
"""
from src.agent.context import ToolContext
from src.agent.agent import run_agent_stream, run_chat_stream
from src.llm.deepseek_client import get_llm
from src.config import DEEPSEEK_MODEL
from src.tracing import trace_span, TraceContext
from langchain_core.messages import HumanMessage, SystemMessage

# ── 意图分类提示词 ──
CLASSIFY_PROMPT = """分析用户输入，判断意图类别。

判定规则:
- ANALYSIS (数据分析): 用户需要对已上传的数据集进行统计、可视化、对比、趋势、分布、
  分组聚合、筛选排序、计算、相关性分析、缺失值处理等操作。
  特征词: 画图、图表、统计、均值、最大值、趋势、对比、分布、占比、相关性、分组、
  排序、筛选、缺失值、清洗、列名、行数、总和、方差、标准差、百分比、直方图、
  散点图、折线图、饼图、热力图、柱状图

- CHAT (闲聊): 问候、常识问答、概念解释、技术咨询、与数据集无关的任意对话。
  特征词: 你好、谢谢、什么是、如何、怎么、为什么、介绍、解释、帮助

只回答一个词: ANALYSIS 或 CHAT"""


def classify_intent(prompt: str) -> str:
    """
    用轻量 LLM 调用判断用户意图

    Args:
        prompt: 用户输入文本

    Returns:
        "analysis" 或 "chat"
    """
    with trace_span("classify_intent"):
        TraceContext.add_meta(model=DEEPSEEK_MODEL)
        llm = get_llm(temperature=0)
        messages = [
            SystemMessage(content=CLASSIFY_PROMPT),
            HumanMessage(content=f"用户输入: {prompt}"),
        ]
        try:
            response = llm.invoke(messages)
            result = str(response.content).strip().upper()
            if "ANALYSIS" in result:
                TraceContext.add_meta(intent="analysis")
                return "analysis"
            TraceContext.add_meta(intent="chat")
            return "chat"
        except Exception:
            TraceContext.add_meta(intent="chat", fallback=True)
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
    with trace_span("route"):
        intent = classify_intent(prompt)
        has_data = ToolContext.has_data()
        TraceContext.add_meta(intent=intent, has_data=has_data)

        if intent == "analysis":
            if not has_data:
                yield {"type": "error", "content": "请先在左侧上传数据文件后再开始分析。"}
                return
            yield from run_agent_stream(prompt, history, temperature)
        else:
            yield from run_chat_stream(prompt, history, temperature)
