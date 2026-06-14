"""
DeepSeek 大模型客户端
─────────────────────
基于 LangChain ChatOpenAI 封装 DeepSeek API 调用。
DeepSeek 兼容 OpenAI 接口格式，通过 base_url 指向 DeepSeek 网关即可。

提供两个核心函数：
  - get_llm(): 获取模型实例，用于 LangChain Agent 链
  - chat():    单轮/多轮对话接口
"""
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.1, streaming: bool = True):
    """
    获取 DeepSeek 大模型实例

    Args:
        temperature: 创造性参数，0=确定性, 1=高随机性
        streaming:   是否启用流式输出（数据分析场景建议关闭）

    Returns:
        ChatOpenAI 实例，已配置为 DeepSeek 兼容端点
    """
    from src.config import DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
        streaming=streaming,
        max_retries=3,
        request_timeout=60,
    )


# ── 系统角色提示词 ──
# 定义数据分析助手的身份和行为准则
SYSTEM_PROMPT = """你是一个专业的数据分析智能助手。你的职责是：
1. 帮助用户理解和分析数据
2. 提供清晰的数据洞察和建议
3. 用中文回答用户的问题
请保持回答简洁、专业、有建设性。"""


def chat(prompt: str, history: list[dict] | None = None) -> str:
    """
    发送消息到 DeepSeek 并返回回复（非流式）

    Args:
        prompt:  用户当前输入文本
        history: 历史对话记录，格式为 [{"role": "user"|"assistant", "content": "..."}, ...]

    Returns:
        模型的文本回复
    """
    # 获取模型实例
    llm = get_llm()

    # 构建消息列表：系统提示 + 历史对话 + 当前用户输入
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # 调用模型
    response = llm.invoke(messages)
    return response.content
