"""
可视化工具模块
───────────────
基于 Matplotlib 和 Seaborn 生成数据分析图表，
图片保存到 outputs/ 目录，返回文件路径供 Streamlit 渲染。

修复记录:
  - 字体设置提前到 seaborn 之前，避免被覆盖
  - 柱状图限制颜色数量，大数据量自动降采样
  - 折线图按 X 轴排序后绘制
  - 饼图超过8类时合并为"其他"
  - 散点图趋势线加 NaN 处理
  - 全局 DPI 提升至 200，字体加大

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
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头渲染，避免 GUI 线程冲突
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from io import BytesIO
from PIL import Image
from src.agent.context import ToolContext
from src.tools.analyzer import _maybe_sample, SAMPLE_MAX_ROWS
from src.config import VIZ_MAX_PIE_CATEGORIES, VIZ_MAX_BAR_CATEGORIES, VIZ_DPI
from src.logger import get_logger

_log = get_logger("visualizer")

# ── 全局样式配置 ──
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 设置 seaborn 样式（必须在字体设置之前，sns.set_style 会重置 font.family）
sns.set_style("whitegrid")

# 3. 设置中文字体（必须在 seaborn 之后，否则被 sns.set_style 重置）
fm._load_fontmanager(try_read_cache=False)

_cn_font_path = None
for f in fm.fontManager.ttflist:
    if "Microsoft YaHei" in f.name and "Light" not in f.name and "UI" not in f.name:
        _cn_font_path = f.fname
        break

if not _cn_font_path:
    for f in fm.fontManager.ttflist:
        if "SimHei" in f.name:
            _cn_font_path = f.fname
            break

if _cn_font_path:
    fm.fontManager.addfont(_cn_font_path)
    _font_prop = fm.FontProperties(fname=_cn_font_path)
    _font_name = _font_prop.get_name()
    plt.rcParams["font.family"] = _font_name
    plt.rcParams["font.sans-serif"] = [_font_name]
    plt.rcParams["axes.unicode_minus"] = False
    _log.info("使用中文字体: %s (%s)", _font_name, _cn_font_path)
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    _log.warning("未找到中文字体，可能显示为方块")

# 3. 防止大数据量渲染时超出 Agg 渲染器路径块限制
matplotlib.rcParams["agg.path.chunksize"] = 20000
matplotlib.rcParams["path.simplify_threshold"] = 0.2

# 4. 全局图表参数优化
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": VIZ_DPI,
    "savefig.dpi": VIZ_DPI,
    "figure.facecolor": "white",
})

# 初始化时清理可能残留的 figure 状态，防止跨调用污染
plt.close("all")

# 饼图最多显示的分类数（超出则合并为"其他"）
MAX_PIE_CATEGORIES = VIZ_MAX_PIE_CATEGORIES
# 柱状图最多显示的条数（超出则取 Top N）
MAX_BAR_CATEGORIES = VIZ_MAX_BAR_CATEGORIES


def _save_fig(name: str) -> str:
    """
    保存当前图表到 outputs/ 目录，做空白图检测 + 去 Alpha 通道处理

    Args:
        name: 图表前缀名（不含扩展名）

    Returns:
        保存的文件绝对路径；若检测为空白图则返回错误信息
    """
    import time
    ts = int(time.time() * 1000)
    filename = f"{name}_{ts}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # 先显式触发 Agg 渲染器绘制，确保图形内容完全就绪
    plt.gcf().canvas.draw()

    # 再保存到内存缓冲区，检测内容后再写入磁盘
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="white", transparent=False,
                pad_inches=0.3)
    plt.close()
    buf.seek(0)

    try:
        img = Image.open(buf)
    except Exception:
        return "错误: 图表渲染失败"

    # RGBA → RGB：去除 Alpha 通道，避免 Streamlit 暗色主题下渲染异常
    if img.mode == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img.close()
        img = rgb_img
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 空白图检测：采样检测，若 >99% 纯白或 >99% 纯黑则视为空白
    # 正常稀疏图表（如折线图）仍有 5-10% 的非白像素（线、文字、网格线）
    w, h = img.size
    total_pixels = w * h
    sample_target = 1000
    sample_step = max(1, total_pixels // sample_target)
    pixels = list(img.getdata())
    white_count = 0
    black_count = 0
    for i in range(0, total_pixels, sample_step):
        r, g, b = pixels[i]
        if r > 250 and g > 250 and b > 250:
            white_count += 1
        elif r < 10 and g < 10 and b < 10:
            black_count += 1
    sampled = (total_pixels + sample_step - 1) // sample_step
    white_ratio = white_count / sampled if sampled else 0
    black_ratio = black_count / sampled if sampled else 0

    if white_ratio > 0.99:
        buf.close()
        return "错误: 图表为空白(全白)，请检查数据是否有效后重试"
    if black_ratio > 0.99:
        buf.close()
        return "错误: 图表为空白(全黑)，请检查数据是否有效后重试"

    # 写入磁盘
    try:
        img.save(filepath, format="PNG")
    except Exception:
        buf.close()
        return "错误: 图表文件写入失败"
    finally:
        img.close()
        buf.close()

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

    # 按 X 轴排序，避免连线错乱
    plot_df = df[[x_column, y_column]].dropna().sort_values(by=x_column)
    if len(plot_df) == 0:
        return "错误: 过滤后数据为空，无法绘制折线图"

    plot_df, was_sampled = _maybe_sample(plot_df)
    title = f"{y_column} 随 {x_column} 变化趋势"
    if was_sampled:
        title += f" (采样数据, n={SAMPLE_MAX_ROWS})"

    plt.close("all")
    try:
        plt.figure(figsize=(10, 5), facecolor="white")
        plt.plot(plot_df[x_column], plot_df[y_column], marker="o", linewidth=2, markersize=4, color="#2196F3")
        plt.title(title, fontsize=16)
        plt.xlabel(x_column, fontsize=13)
        plt.ylabel(y_column, fontsize=13)
        plt.xticks(rotation=45)
        path = _save_fig("line_chart")
        return path
    finally:
        plt.close("all")


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

    # 聚合计算：按 x 列分组求 y 列均值，取 Top N
    work_df, was_sampled = _maybe_sample(df)
    agg_df = work_df.groupby(x_column)[y_column].mean().sort_values(ascending=False)
    if len(agg_df) == 0:
        return "错误: 分组聚合后数据为空，无法绘制柱状图"

    # 分类太多时只取 Top N
    if len(agg_df) > MAX_BAR_CATEGORIES:
        agg_df = agg_df.head(MAX_BAR_CATEGORIES)

    colors = sns.color_palette("Blues_d", len(agg_df))

    title = f"{y_column} 按 {x_column} 分布"
    if was_sampled:
        title += f" (采样数据, n={SAMPLE_MAX_ROWS})"

    plt.close("all")
    try:
        plt.figure(figsize=(10, 5), facecolor="white")
        plt.bar(agg_df.index.astype(str), agg_df.values, color=colors, edgecolor="white", linewidth=0.5)
        plt.title(title, fontsize=16)
        plt.xlabel(x_column, fontsize=13)
        plt.ylabel(y_column, fontsize=13)
        plt.xticks(rotation=45)
        path = _save_fig("bar_chart")
        return path
    finally:
        plt.close("all")


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

    # 去除 NaN，避免 polyfit 崩溃
    plot_df = df[[x_column, y_column]].dropna()
    if len(plot_df) == 0:
        return "错误: 数据列全为空值，无法绘图"

    plot_df, was_sampled = _maybe_sample(plot_df)
    title = f"{y_column} vs {x_column}"
    if was_sampled:
        title += f" (采样数据, n={SAMPLE_MAX_ROWS})"

    plt.close("all")
    try:
        plt.figure(figsize=(10, 6), facecolor="white")
        plt.scatter(plot_df[x_column], plot_df[y_column], alpha=0.5, edgecolors="white", s=50, color="#2196F3")
        plt.title(title, fontsize=16)
        plt.xlabel(x_column, fontsize=13)
        plt.ylabel(y_column, fontsize=13)

        # 添加趋势线（确保数据点 ≥2 且无非 NaN）
        if len(plot_df) >= 2:
            try:
                x_vals = plot_df[x_column].values.astype(float)
                y_vals = plot_df[y_column].values.astype(float)
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                plt.plot(x_line, p(x_line), "r--", linewidth=2, label="趋势线")
                plt.legend(fontsize=11)
            except Exception:
                pass

        path = _save_fig("scatter_plot")
        return path
    finally:
        plt.close("all")


def pie_chart(column: str) -> str:
    """
    饼图 — 适合展示各类别的占比

    超过 MAX_PIE_CATEGORIES 个分类时，自动将小类合并为"其他"。

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

    work_df, was_sampled = _maybe_sample(df)
    counts = work_df[column].value_counts()
    if len(counts) == 0:
        return "错误: 指定列无有效数据，无法绘制饼图"

    # 分类过多时合并小类为"其他"
    if len(counts) > MAX_PIE_CATEGORIES:
        top = counts.head(MAX_PIE_CATEGORIES - 1)
        others = pd.Series({"其他": counts.iloc[MAX_PIE_CATEGORIES - 1:].sum()})
        counts = pd.concat([top, others])

    colors = sns.color_palette("pastel", len(counts))

    title = f"{column} 分布占比"
    if was_sampled:
        title += f" (采样数据, n={SAMPLE_MAX_ROWS})"

    plt.close("all")
    try:
        plt.figure(figsize=(8, 8), facecolor="white")
        wedges, texts, autotexts = plt.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            pctdistance=0.75,
        )
        # 标签字体加大
        for t in texts:
            t.set_fontsize(11)
        for t in autotexts:
            t.set_fontsize(10)
            t.set_color("white")

        plt.title(title, fontsize=16)

        path = _save_fig("pie_chart")
        return path
    finally:
        plt.close("all")


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

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
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

    target, was_sampled = _maybe_sample(target)
    corr = target.corr()

    title = "相关性热力图"
    if was_sampled:
        title += f" (采样数据, n={SAMPLE_MAX_ROWS})"

    plt.close("all")
    try:
        plt.figure(figsize=(10, 8), facecolor="white")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, linewidths=0.5,
                    annot_kws={"fontsize": 10})
        plt.title(title, fontsize=16)

        path = _save_fig("heatmap")
        return path
    finally:
        plt.close("all")


def histogram(column: str, bins: int = 20) -> str:
    """
    直方图 — 展示数值列的分布形态

    Args:
        column: 数值列名
        bins:   分箱数量，默认 20

    Returns:
        图表文件路径
    """
    df = ToolContext.get()
    if df is None:
        return "错误: 没有加载任何数据"
    if column not in df.columns:
        return f"错误: 列 '{column}' 不存在"

    data = df[column].dropna()
    if len(data) == 0:
        return "错误: 指定列全为空值"

    data, was_sampled = _maybe_sample(data)
    title = f"{column} 分布直方图"
    if was_sampled:
        title += f" (采样数据, n={SAMPLE_MAX_ROWS})"

    plt.close("all")
    try:
        plt.figure(figsize=(10, 5), facecolor="white")
        plt.hist(data, bins=bins, color="#2196F3", edgecolor="white", alpha=0.85)
        plt.title(title, fontsize=16)
        plt.xlabel(column, fontsize=13)
        plt.ylabel("频次", fontsize=13)

        # 添加均值线
        mean_val = data.mean()
        plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
                    label=f"均值: {mean_val:.2f}")
        plt.legend(fontsize=11)

        path = _save_fig("histogram")
        return path
    finally:
        plt.close("all")
