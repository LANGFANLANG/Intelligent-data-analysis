"""
统计分析工具模块
─────────────────
提供描述性统计、相关性分析、分组聚合、频率统计等数据分析功能。
所有函数通过 ToolContext 获取当前 DataFrame，返回文本或表格结果。

工具列表:
  describe_data()     - 描述性统计 (count/mean/std/min/25%/50%/75%/max)
  correlate()         - 计算数值列之间的相关系数矩阵
  groupby_agg()       - 分组聚合 (sum/mean/count/min/max/std)
  value_counts()      - 单列频率统计 (Top N)
"""
import pandas as pd
from src.agent.context import ToolContext


def describe_data(columns: str = "") -> str:
    """
    对数据框执行描述性统计

    Args:
        columns: 指定列名（逗号分隔），留空则分析所有列

    Returns:
        各列的 count/mean/std/min/25%/50%/75%/max 统计表
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    # 筛选列
    if columns:
        cols = [c.strip() for c in columns.split(",")]
        valid = [c for c in cols if c in df.columns]
        if not valid:
            return f"错误: 指定的列 {cols} 都不存在"
        target = df[valid]
    else:
        # 默认分析所有数值列
        target = df.select_dtypes(include="number")
        if target.empty:
            target = df

    # 生成描述性统计
    desc = target.describe()
    return desc.to_string()


def correlate(columns: str = "") -> str:
    """
    计算数值列之间的皮尔逊相关系数矩阵

    Args:
        columns: 指定列名（逗号分隔），留空则计算所有数值列的相关性

    Returns:
        相关系数矩阵（介于 -1 到 1 之间）
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return "数据中没有数值列，无法计算相关性"

    # 筛选指定列
    if columns:
        cols = [c.strip() for c in columns.split(",")]
        valid = [c for c in cols if c in numeric.columns]
        if not valid:
            return f"错误: 指定的列 {cols} 中没有有效的数值列"
        target = numeric[valid]
    else:
        target = numeric

    if target.shape[1] < 2:
        return "错误: 至少需要2个数值列才能计算相关性"

    corr_matrix = target.corr()
    return corr_matrix.to_string()


def groupby_agg(group_col: str, value_col: str, agg_func: str = "mean") -> str:
    """
    按指定列分组后对目标列执行聚合计算

    Args:
        group_col:  分组依据列名
        value_col:  聚合目标列名
        agg_func:   聚合函数 (mean/sum/count/min/max/std/median)

    Returns:
        分组聚合结果表格
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    if group_col not in df.columns:
        return f"错误: 分组列 '{group_col}' 不存在"
    if value_col not in df.columns:
        return f"错误: 目标列 '{value_col}' 不存在"

    valid_funcs = {"mean", "sum", "count", "min", "max", "std", "median"}
    if agg_func not in valid_funcs:
        return f"错误: 不支持的聚合函数 '{agg_func}'，支持: {', '.join(valid_funcs)}"

    result = df.groupby(group_col)[value_col].agg(agg_func)
    return result.to_string()


def value_counts(column: str, top_n: int = 10) -> str:
    """
    统计指定列中各值的出现频率（Top N）

    Args:
        column: 目标列名
        top_n:  返回前 N 个最多的值

    Returns:
        频率统计表（按频次降序）
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    if column not in df.columns:
        return f"错误: 列 '{column}' 不存在"

    counts = df[column].value_counts().head(top_n)
    return counts.to_string()
