"""
可视化工具模块
───────────────
基于 Matplotlib 和 Seaborn 生成数据分析图表，
图片保存到 outputs/ 目录，返回文件路径供 Streamlit 渲染。

图表样式统一使用中文字体适配，每次生成前自动更新数据上下文。

工具列表:
  line_chart()    - 折线图（趋势分析）
  bar_chart()     - 柱状图（分类对比）
  scatter_plot()  - 散点图（相关性可视化）
  pie_chart()     - 饼图（占比分布）
  heatmap()       - 热力图（相关性矩阵可视化）
  histogram()     - 直方图（分布分析）
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头渲染，避免 GUI 线程冲突
import matplotlib.pyplot as plt
import seaborn as sns
from src.agent.context import ToolContext

# ── 全局样式配置 ──
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# 设置中文字体，确保图表中文正常显示
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def _save_fig(name: str) -> str:
    """
    保存当前图表到 outputs/ 目录

    Args:
        name: 图表前缀名（不含扩展名）

    Returns:
        保存的文件绝对路径，文件名格式: outputs/{name}_{timestamp}.png
    """
    import time
    ts = int(time.time() * 1000)
    filename = f"{name}_{ts}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


def line_chart(x_column: str, y_column: str) -> str:
    """
    折线图 — 适合展示随时间变化的趋势

    Args:
        x_column: X 轴列名
        y_column: Y 轴列名

    Returns:
        图表文件路径
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if x_column not in df.columns or y_column not in df.columns:
        return f"错误: 列 '{x_column}' 或 '{y_column}' 不存在"

    plt.figure(figsize=(10, 5))
    plt.plot(df[x_column], df[y_column], marker="o", linewidth=2, markersize=4)
    plt.title(f"{y_column} 随 {x_column} 变化趋势", fontsize=14)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xticks(rotation=45)

    path = _save_fig("line_chart")
    return path


def bar_chart(x_column: str, y_column: str) -> str:
    """
    柱状图 — 适合分类对比

    Args:
        x_column: X 轴列名（分类列）
        y_column: Y 轴列名（数值列）

    Returns:
        图表文件路径
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if x_column not in df.columns or y_column not in df.columns:
        return f"错误: 列 '{x_column}' 或 '{y_column}' 不存在"

    plt.figure(figsize=(10, 5))
    plt.bar(df[x_column].astype(str), df[y_column], color=sns.color_palette("viridis", len(df)))
    plt.title(f"{y_column} 按 {x_column} 分布", fontsize=14)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xticks(rotation=45)

    path = _save_fig("bar_chart")
    return path


def scatter_plot(x_column: str, y_column: str) -> str:
    """
    散点图 — 适合展示两个数值变量之间的关系

    Args:
        x_column: X 轴列名（数值列）
        y_column: Y 轴列名（数值列）

    Returns:
        图表文件路径
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if x_column not in df.columns or y_column not in df.columns:
        return f"错误: 列 '{x_column}' 或 '{y_column}' 不存在"

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.6, edgecolors="white", s=60)
    plt.title(f"{y_column} vs {x_column}", fontsize=14)
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    # 添加趋势线
    try:
        from numpy import polyfit
        z = polyfit(df[x_column], df[y_column], 1)
        p = __import__("numpy").poly1d(z)
        x_sorted = sorted(df[x_column])
        plt.plot(x_sorted, p(x_sorted), "r--", linewidth=2, label="趋势线")
        plt.legend()
    except Exception:
        pass

    path = _save_fig("scatter_plot")
    return path


def pie_chart(column: str) -> str:
    """
    饼图 — 适合展示各类别的占比

    Args:
        column: 分类列名

    Returns:
        图表文件路径
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if column not in df.columns:
        return f"错误: 列 '{column}' 不存在"

    counts = df[column].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("pastel"), startangle=90)
    plt.title(f"{column} 分布占比", fontsize=14)

    path = _save_fig("pie_chart")
    return path


def heatmap(columns: str = "") -> str:
    """
    热力图 — 可视化数值列之间的相关性矩阵

    Args:
        columns: 指定列名（逗号分隔），留空则使用全部数值列

    Returns:
        图表文件路径
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"

    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return "数据中没有数值列，无法生成热力图"

    if columns:
        cols = [c.strip() for c in columns.split(",")]
        valid = [c for c in cols if c in numeric.columns]
        if not valid:
            return f"错误: 指定的列 {cols} 中没有有效的数值列"
        target = numeric[valid]
    else:
        target = numeric

    if target.shape[1] < 2:
        return "错误: 至少需要2个数值列才能生成热力图"

    plt.figure(figsize=(10, 8))
    corr = target.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5)
    plt.title("相关性热力图", fontsize=14)

    path = _save_fig("heatmap")
    return path


def histogram(column: str, bins: int = 20) -> str:
    """
    直方图 — 展示数值列的分布形态

    Args:
        column: 数值列名
        bins:   分箱数量

    Returns:
        图表文件路径
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if column not in df.columns:
        return f"错误: 列 '{column}' 不存在"

    plt.figure(figsize=(10, 5))
    plt.hist(df[column].dropna(), bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    plt.title(f"{column} 分布直方图", fontsize=14)
    plt.xlabel(column)
    plt.ylabel("频次")

    path = _save_fig("histogram")
    return path
