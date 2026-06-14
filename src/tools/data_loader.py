"""
数据加载工具模块
─────────────────
提供从用户上传文件中加载 DataFrame 的能力。
支持 CSV、Excel (xlsx/xls)、JSON 三种格式。

使用 Streamlit 的 @st.cache_data 缓存解析结果，
避免每次页面刷新时重复解析大文件。
"""
import os
import pandas as pd
import streamlit as st
from io import StringIO
from src.config import MAX_FILE_SIZE_MB as _MAX_FILE_SIZE_MB

MAX_FILE_SIZE = _MAX_FILE_SIZE_MB * 1024 * 1024


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame | None:
    """
    解析 CSV 文件为 DataFrame，自动尝试 utf-8 → gbk 编码

    Args:
        file: Streamlit UploadedFile 对象

    Returns:
        DataFrame 或 None（解析失败时）
    """
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            content = file.getvalue()
            return pd.read_csv(StringIO(content.decode(encoding, errors="replace")))
        except UnicodeDecodeError:
            file.seek(0)
            continue
        except Exception as e:
            st.error(f"CSV 加载失败: {e}")
            return None
    st.error("CSV 编码无法识别，请转换为 UTF-8 后重试")
    return None


@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame | None:
    """
    解析 Excel 文件为 DataFrame

    使用 openpyxl 引擎以避免依赖 Microsoft Excel 程序。

    Args:
        file: Streamlit UploadedFile 对象

    Returns:
        DataFrame 或 None（解析失败时）
    """
    try:
        return pd.read_excel(file, engine="openpyxl")
    except Exception as e:
        st.error(f"Excel 加载失败: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_json(file) -> pd.DataFrame | None:
    """
    解析 JSON 文件为 DataFrame

    先将 bytes 解码为 UTF-8 字符串，再通过 StringIO 传入 pandas。

    Args:
        file: Streamlit UploadedFile 对象

    Returns:
        DataFrame 或 None（解析失败时）
    """
    try:
        content = file.getvalue().decode("utf-8")
        return pd.read_json(StringIO(content))
    except Exception as e:
        st.error(f"JSON 加载失败: {e}")
        return None


def load_file(uploaded_file) -> pd.DataFrame | None:
    """
    根据文件扩展名自动选择加载器

    校验流程:
      1. 文件大小检查（超过上限则拒绝）
      2. 扩展名匹配到对应解析器
      3. 解析后检查 DataFrame 是否为空

    Args:
        uploaded_file: Streamlit UploadedFile 对象

    Returns:
        DataFrame 或 None（不合规 / 解析失败 / 空数据时）
    """
    # 文件大小校验
    file_size = uploaded_file.size if hasattr(uploaded_file, "size") else 0
    if file_size > MAX_FILE_SIZE:
        st.error(
            f"文件过大（{file_size / 1024 / 1024:.1f}MB），"
            f"当前上限 {_MAX_FILE_SIZE_MB}MB。请拆分文件或调整 MAX_FILE_SIZE_MB 环境变量"
        )
        return None
    if file_size == 0:
        st.error("文件为空，请上传有效数据文件")
        return None

    # 扩展名路由
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = load_csv(uploaded_file)
    elif filename.endswith((".xls", ".xlsx")):
        df = load_excel(uploaded_file)
    elif filename.endswith(".json"):
        df = load_json(uploaded_file)
    else:
        st.error(f"不支持的文件格式: {uploaded_file.name}")
        return None

    # 空 DataFrame 检测
    if df is not None and df.empty:
        st.error("文件已成功解析，但数据为空（0行）")
        return None

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    生成 DataFrame 的摘要信息

    用于在侧边栏展示数据概览，包含维度、列信息、缺失值、统计描述等。

    Args:
        df: 已加载的 DataFrame

    Returns:
        摘要字典: {shape, columns, dtypes, missing, describe, head}
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe(include="all"),
        "head": df.head(100),
    }
