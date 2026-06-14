"""
数据清洗工具模块
─────────────────
提供常见的数据预处理操作，包括缺失值处理、去重、数据类型转换等。
所有函数通过 ToolContext 获取当前 DataFrame，修改后写回。

工具列表:
  check_missing()        - 检查缺失值
  fill_missing_mean()    - 均值填充
  fill_missing_median()  - 中位数填充
  fill_missing_mode()    - 众数填充
  drop_missing()         - 删除含缺失值的行
  drop_duplicates()      - 删除重复行
  drop_column()          - 删除指定列
"""
import pandas as pd
from src.agent.context import ToolContext
from src.tools.analyzer import _maybe_sample, SAMPLE_MAX_ROWS


def check_missing() -> str:
    """
    检查当前 DataFrame 中每列的缺失值数量

    Returns:
        缺失值统计报告文本
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    missing = df.isnull().sum()
    total = len(df)
    # 筛选出有缺失值的列
    missing = missing[missing > 0]

    if missing.empty:
        return "数据完整，没有缺失值"

    lines = [f"缺失值统计 (共 {total} 行):"]
    for col, count in missing.items():
        pct = count / total * 100
        lines.append(f"  - {col}: {count} 个缺失 ({pct:.1f}%)")
    return "\n".join(lines)


def fill_missing_mean(column: str) -> str:
    """
    用列均值填充指定列的缺失值

    Args:
        column: 需要填充的列名（必须是数值列）

    Returns:
        操作结果描述
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if column not in df.columns:
        return f"错误: 列 '{column}' 不存在"

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"错误: 列 '{column}' 不是数值类型，无法用均值填充"

    before = df[column].isnull().sum()
    sample_data, was_sampled = _maybe_sample(df[column].dropna())
    fill_val = sample_data.mean()
    df[column].fillna(fill_val, inplace=True)
    msg = f"已用均值 {fill_val:.4f} 填充列 '{column}' 的 {before} 个缺失值"
    if was_sampled:
        msg += f" (基于 {SAMPLE_MAX_ROWS} 条采样数据估算)"
    return msg


def fill_missing_median(column: str) -> str:
    """
    用列中位数填充指定列的缺失值

    Args:
        column: 需要填充的列名（必须是数值列）

    Returns:
        操作结果描述
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if column not in df.columns:
        return f"错误: 列 '{column}' 不存在"

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"错误: 列 '{column}' 不是数值类型，无法用中位数填充"

    before = df[column].isnull().sum()
    sample_data, was_sampled = _maybe_sample(df[column].dropna())
    fill_val = sample_data.median()
    df[column].fillna(fill_val, inplace=True)
    msg = f"已用中位数 {fill_val:.4f} 填充列 '{column}' 的 {before} 个缺失值"
    if was_sampled:
        msg += f" (基于 {SAMPLE_MAX_ROWS} 条采样数据估算)"
    return msg


def fill_missing_mode(column: str) -> str:
    """
    用列众数填充指定列的缺失值

    Args:
        column: 需要填充的列名

    Returns:
        操作结果描述
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if column not in df.columns:
        return f"错误: 列 '{column}' 不存在"

    before = df[column].isnull().sum()
    sample_data, was_sampled = _maybe_sample(df[column].dropna())
    mode_val = sample_data.mode()
    if len(mode_val) > 0:
        df[column].fillna(mode_val[0], inplace=True)
        msg = f"已用众数 '{mode_val[0]}' 填充列 '{column}' 的 {before} 个缺失值"
        if was_sampled:
            msg += f" (基于 {SAMPLE_MAX_ROWS} 条采样数据估算)"
        return msg
    return f"列 '{column}' 没有众数，填充失败"


def drop_missing(column: str | None = None) -> str:
    """
    删除含缺失值的行

    Args:
        column: 指定列名则只检查该列，不指定则检查所有列

    Returns:
        操作结果描述
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    before = len(df)
    if column:
        if column not in df.columns:
            return f"错误: 列 '{column}' 不存在"
        df.dropna(subset=[column], inplace=True)
    else:
        df.dropna(inplace=True)

    removed = before - len(df)
    return f"删除了 {removed} 行包含缺失值的数据，剩余 {len(df)} 行"


def drop_duplicates() -> str:
    """
    删除完全重复的行

    Returns:
        操作结果描述
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    return f"删除了 {removed} 行重复数据，剩余 {len(df)} 行"


def drop_column(columns: str) -> str:
    """
    删除指定列

    Args:
        columns: 要删除的列名，多个列用逗号分隔

    Returns:
        操作结果描述
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    cols_to_drop = [c.strip() for c in columns.split(",")]
    valid = [c for c in cols_to_drop if c in df.columns]
    invalid = [c for c in cols_to_drop if c not in df.columns]

    if not valid:
        return f"错误: 指定的列 {cols_to_drop} 都不存在"

    df.drop(columns=valid, inplace=True)
    msg = f"已删除列: {', '.join(valid)}"
    if invalid:
        msg += f"，以下列不存在: {', '.join(invalid)}"
    return msg
