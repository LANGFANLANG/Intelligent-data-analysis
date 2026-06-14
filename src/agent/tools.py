"""
LangChain Tool 注册表
─────────────────────
将 tools/ 下的所有分析工具函数包装为 LangChain 可调用的 Tool 对象，
供 ReAct Agent 在思考-行动-观察循环中使用。

每个 Tool 的 docstring 会被作为描述传递给 LLM，
帮助 Agent 理解何时以及如何调用该工具。
"""
from langchain.tools import tool
from datetime import datetime, timezone

# ── 数据清洗工具 ──
from src.tools.data_cleaner import (
    check_missing,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
    drop_missing,
    drop_duplicates,
    drop_column,
)

# ── 统计分析工具 ──
from src.tools.analyzer import (
    describe_data,
    correlate,
    groupby_agg,
    value_counts,
)

# ── 可视化工具 ──
from src.tools.visualizer import (
    line_chart,
    bar_chart,
    scatter_plot,
    pie_chart,
    heatmap,
    histogram,
)

# ── 工具注册 ──
# 每个 @tool 包裹 try/except，确保工具异常不会导致 Agent 崩溃，
# 而是以错误字符串的形式返回给 Agent，Agent 可以据此调整策略


@tool
def tool_check_missing() -> str:
    """
    检查数据框中每列的缺失值数量。
    在开始任何分析之前，建议先调用此工具了解数据质量。
    无需参数。
    """
    try:
        return check_missing()
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_fill_missing_mean(column: str) -> str:
    """
    用该列的均值填充指定数值列的缺失值。
    适用于数据近似正态分布的场景。

    Args:
        column: 需要填充的数值列名
    """
    try:
        return fill_missing_mean(column)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_fill_missing_median(column: str) -> str:
    """
    用该列的中位数填充指定数值列的缺失值。
    适用于数据存在极端值或偏态分布的场景。

    Args:
        column: 需要填充的数值列名
    """
    try:
        return fill_missing_median(column)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_fill_missing_mode(column: str) -> str:
    """
    用该列的众数填充指定列的缺失值。
    适用于分类数据或离散值填充。

    Args:
        column: 需要填充的列名
    """
    try:
        return fill_missing_mode(column)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_drop_missing(column: str = "") -> str:
    """
    删除包含缺失值的行。
    如果指定列名则只检查该列，否则检查所有列。

    Args:
        column: 可选，指定要检查的列名，留空检查所有列
    """
    try:
        return drop_missing(column if column else None)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_drop_duplicates() -> str:
    """
    删除数据框中完全重复的行。
    无需参数。
    """
    try:
        return drop_duplicates()
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_drop_column(columns: str) -> str:
    """
    删除指定的列。多个列名用逗号分隔。

    Args:
        columns: 要删除的列名，多个用逗号分隔，例如 "col1,col2"
    """
    try:
        return drop_column(columns)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_describe_data(columns: str = "") -> str:
    """
    对数据执行描述性统计，返回 count/mean/std/min/25%/50%/75%/max 等信息。
    在开始分析前，建议先用此工具了解数据分布。

    Args:
        columns: 可选，指定列名（逗号分隔），留空分析所有数值列
    """
    try:
        return describe_data(columns if columns else "")
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_correlate(columns: str = "") -> str:
    """
    计算数值列之间的皮尔逊相关系数矩阵（-1到1）。
    用于发现变量之间的线性关系。越接近1或-1表示相关性越强。

    Args:
        columns: 可选，指定要分析的列名（逗号分隔），留空分析所有数值列
    """
    try:
        return correlate(columns if columns else "")
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_groupby_agg(group_col: str, value_col: str, agg_func: str = "mean") -> str:
    """
    按指定列分组后对目标列进行聚合计算。
    例如：按'地区'分组计算'销售额'的平均值。

    Args:
        group_col: 分组依据列名
        value_col: 聚合目标列名
        agg_func:  聚合函数，可选 mean/sum/count/min/max/std/median，默认 mean
    """
    try:
        return groupby_agg(group_col, value_col, agg_func)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_value_counts(column: str, top_n: int = 10) -> str:
    """
    统计指定列中各值的出现频次，返回 Top N。
    用于了解分类变量的分布情况。

    Args:
        column: 目标列名
        top_n:  返回前 N 个最多的值，默认 10
    """
    try:
        return value_counts(column, top_n)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_line_chart(x_column: str, y_column: str) -> str:
    """
    生成折线图，适合展示数据随时间变化的趋势。
    X 轴通常为时间或有序类别列。
    返回图表文件路径。

    Args:
        x_column: X 轴列名（时间/有序类别）
        y_column: Y 轴列名（数值）
    """
    try:
        return line_chart(x_column, y_column)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_bar_chart(x_column: str, y_column: str) -> str:
    """
    生成柱状图，适合展示不同类别之间的数值对比。
    返回图表文件路径。

    Args:
        x_column: X 轴列名（分类列）
        y_column: Y 轴列名（数值列）
    """
    try:
        return bar_chart(x_column, y_column)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_scatter_plot(x_column: str, y_column: str) -> str:
    """
    生成散点图，适合展示两个数值变量之间的关系。
    会自动添加红色趋势线。
    返回图表文件路径。

    Args:
        x_column: X 轴列名（数值列）
        y_column: Y 轴列名（数值列）
    """
    try:
        return scatter_plot(x_column, y_column)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_pie_chart(column: str) -> str:
    """
    生成饼图，展示各类别在整体中的占比。
    返回图表文件路径。

    Args:
        column: 分类列名
    """
    try:
        return pie_chart(column)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_heatmap(columns: str = "") -> str:
    """
    生成相关性热力图，用颜色深浅展示数值列之间的相关程度。
    红=正相关，蓝=负相关，白=无相关。
    返回图表文件路径。

    Args:
        columns: 可选，指定列名（逗号分隔），留空分析所有数值列
    """
    try:
        return heatmap(columns if columns else "")
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_histogram(column: str, bins: int = 20) -> str:
    """
    生成直方图，展示数值列的数据分布形态。
    用于判断数据是否服从正态分布、是否存在偏态等。
    返回图表文件路径。

    Args:
        column: 数值列名
        bins:   分箱数量，默认 20
    """
    try:
        return histogram(column, bins)
    except Exception as e:
        return f"工具执行异常: {e}"


@tool
def tool_get_current_time() -> str:
    """
    获取当前日期和时间。
    当用户提到"今天/当前/最近/本周/本月"等时间相关问题时，
    必须先调用此工具获取准确日期，再结合数据进行筛选分析。
    无需参数。

    Returns:
        当前北京时间 + UTC 时间字符串
    """
    try:
        from datetime import timedelta
        now_utc = datetime.now(timezone.utc)
        beijing_tz = timezone(timedelta(hours=8))
        now_beijing = now_utc.astimezone(beijing_tz)
        weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
        weekday = weekday_map[now_beijing.weekday()]
        return (
            f"北京时间: {now_beijing.strftime('%Y-%m-%d')} 星期{weekday} {now_beijing.strftime('%H:%M:%S')} (UTC+8), "
            f"UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        return f"工具执行异常: {e}"


# ── 全部工具列表 ──
ALL_TOOLS = [
    # 数据清洗
    tool_check_missing,
    tool_fill_missing_mean,
    tool_fill_missing_median,
    tool_fill_missing_mode,
    tool_drop_missing,
    tool_drop_duplicates,
    tool_drop_column,
    # 统计分析
    tool_describe_data,
    tool_correlate,
    tool_groupby_agg,
    tool_value_counts,
    # 可视化
    tool_line_chart,
    tool_bar_chart,
    tool_scatter_plot,
    tool_pie_chart,
    tool_heatmap,
    tool_histogram,
    # 工具
    tool_get_current_time,
]
