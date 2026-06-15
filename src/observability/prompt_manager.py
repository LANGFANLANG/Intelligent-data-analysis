"""
Prompt 版本管理模块
──────────────────
通过 Langfuse Prompt Management 实现系统提示词的版本化托管。

功能:
  - 从 Langfuse 拉取最新版 Prompt（支持标签如 "production"）
  - 本地 fallback：Langfuse 不可用时使用代码中的默认值
  - 缓存机制：按标签缓存，避免每次请求都远程拉取

Prompt 命名约定:
  - datamate-agent-system    → Agent 系统提示词
  - datamate-router-system   → 意图分类提示词
  - datamate-chat-system     → 聊天 Agent 系统提示词

用法:
  from src.observability.prompt_manager import get_prompt, DEFAULT_PROMPTS

  prompt = get_prompt("datamate-agent-system")
  # 返回 Langfuse 托管的最新版，或 fallback 到 DEFAULT_PROMPTS["agent"]
"""
import time
import threading
import logging
from typing import Any

from src.observability.langfuse_client import get_langfuse
from src.logger import get_logger

_log = get_logger("observability.prompt")

# 抑制 Langfuse SDK 的 Prompt 404 ERROR 日志（未创建的 Prompt 用本地 fallback，不需要 ERROR）
logging.getLogger("langfuse").setLevel(logging.WARNING)

# ── 默认 Prompt（Langfuse 不可用时的 fallback）──

DEFAULT_PROMPTS = {
    "datamate-agent-system": """你是一个专业的数据分析 Agent。用户已上传数据文件，你需要通过调用工具来完成分析任务。

工作准则:
1. **先探索后分析**: 先用 describe_data 和 check_missing 了解数据结构
2. **按需可视化**: 只在用户明确要求"画图/可视化/趋势/对比图/分布"等时才调用图表工具
3. **简单问题直达**: 计数、均值、排序、分组统计等简单问题直接用文字工具回答，不画图
4. **中文输出**: 所有分析结论必须用中文表达
5. **洞察导向**: 不只罗列数字，要给出业务洞察和建议
6. **图表配文字**: 如果生成了图表，必须用文字解释图表反映的信息
7. **出错自查**: 如果工具返回"列不存在"错误，先用 describe_data 查看可用列名，再用正确的列名重试

分析报告格式:
- 先总结关键发现
- 再展示分析过程和数据
- 最后给出可操作的建议（如有）""",

    "datamate-db-agent-system": """你是一个专业的数据分析助手，负责分析数据库中的业务数据。

内部工作准则（这些你可以做，但不要告诉用户）:
1. 先用 list_tables 了解有哪些表，再用 describe_table 查看目标表的结构
2. 根据表结构生成正确的 SQL 查询，注意 JOIN 条件、日期格式、字符串引号
3. 使用 MySQL 兼容的 SQL 语法。日期用 'YYYY-MM-DD' 格式，字符串用单引号。
4. 先简单验证数据，确认正确后再写复杂查询
5. sql_query 执行后，结果会自动加载，然后可以用 describe_data / bar_chart 等工具分析
6. 图表配文字: 生成了图表必须用文字解释
7. 出错自查: SQL 报错时先用 describe_table 确认列名，修正后重试

输出规则（必须严格遵守）:
1. **禁止泄露内部信息**: 绝对不能在回复中出现表名、列名、字段名、SQL 语句
2. **禁止描述分析过程**: 不要说"我先查询了..."、"我查看了表结构..."
3. **只用业务术语**: 用"商户"而非"merchants表"，用"订单金额"而非"amount列"
4. **图表在文末**: 如果生成了图表，将图表说明整合到报告中，图表自动显示在下方

输出格式（结构化报告）:
## 摘要
（1-2句话概括核心结论）
## 关键发现
（分点列出重要数据洞察，每条带具体数字支撑）
## 数据概览
（如有必要，列出关键排名或对比数据）
## 建议
（基于分析的可操作建议；如不适用则省略）""",

    "datamate-router-system": """分析用户输入，判断意图类别。

判定规则:
- ANALYSIS (数据分析): 用户需要对已上传的数据集进行统计、可视化、对比、趋势、分布、
  分组聚合、筛选排序、计算、相关性分析、缺失值处理等操作。
  特征词: 画图、图表、统计、均值、最大值、趋势、对比、分布、占比、相关性、分组、
  排序、筛选、缺失值、清洗、列名、行数、总和、方差、标准差、百分比、直方图、
  散点图、折线图、饼图、热力图、柱状图

- CHAT (闲聊): 问候、常识问答、概念解释、技术咨询、与数据集无关的任意对话。
  特征词: 你好、谢谢、什么是、如何、怎么、为什么、介绍、解释、帮助

只回答一个词: ANALYSIS 或 CHAT""",

    "datamate-chat-system": """你是一个友好的 AI 助手 DataMate，擅长数据分析相关的技术问答和日常对话。

你可以:
- 回答数据分析、统计学、机器学习相关的概念问题
- 解释图表、数据指标的含义
- 提供数据分析的方法论建议
- 进行日常闲聊、问候

当用户需要分析具体数据时，引导他们先在左侧上传数据文件。

请用中文回答，语气友好、专业。""",
}


class PromptManager:
    """
    Prompt 管理器

    功能:
      - 从 Langfuse 拉取 Prompt（支持标签筛选）
      - 本地缓存 + fallback 机制
      - 线程安全的缓存操作
    """

    # 缓存结构: {prompt_name: (content, expire_time)}
    _cache: dict[str, tuple[str, float]] = {}
    _lock = threading.Lock()
    # 缓存有效期（秒），默认 5 分钟
    CACHE_TTL = 300

    @classmethod
    def get(cls, prompt_name: str, label: str = "production", fallback: str = "") -> str:
        """
        获取 Prompt 内容

        优先级: Langfuse 远程 > 本地缓存 > fallback 参数 > DEFAULT_PROMPTS

        Args:
            prompt_name: Prompt 名称（如 "datamate-agent-system"）
            label:       标签（如 "production", "latest"）
            fallback:    自定义 fallback 文本（为空则使用 DEFAULT_PROMPTS）

        Returns:
            Prompt 文本内容
        """
        # 1. 检查本地缓存
        with cls._lock:
            if prompt_name in cls._cache:
                content, expire = cls._cache[prompt_name]
                if time.time() < expire:
                    return content

        # 2. 尝试从 Langfuse 拉取
        client = get_langfuse()
        if client is not None:
            try:
                prompt = client.get_prompt(prompt_name, label=label)
                content = prompt.compile()
                # 更新缓存
                with cls._lock:
                    cls._cache[prompt_name] = (content, time.time() + cls.CACHE_TTL)
                _log.debug("从 Langfuse 获取 Prompt: %s (label=%s)", prompt_name, label)
                return content
            except Exception as e:
                _log.debug("从 Langfuse 获取 Prompt 失败 (%s): %s，使用 fallback", prompt_name, e)

        # 3. Fallback
        if fallback:
            return fallback

        default = DEFAULT_PROMPTS.get(prompt_name, "")
        if default:
            _log.debug("使用本地默认 Prompt: %s", prompt_name)
        return default

    @classmethod
    def clear_cache(cls):
        """清空 Prompt 缓存（测试 / Prompt 更新后调用）"""
        with cls._lock:
            cls._cache.clear()
            _log.debug("Prompt 缓存已清空")


def get_prompt(prompt_name: str, label: str = "production") -> str:
    """
    便捷函数: 获取 Prompt 内容
    """
    return PromptManager.get(prompt_name, label=label)
