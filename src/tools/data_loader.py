"""
数据加载工具模块
─────────────────
提供从用户上传文件中加载 DataFrame 的能力。
支持 CSV、Excel (xlsx/xls)、JSON 三种格式。

使用 Streamlit 的 @st.cache_data 缓存解析结果，
避免每次页面刷新时重复解析大文件。
"""
import pandas as pd
import streamlit as st
from io import StringIO


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame | None:
    """
    解析 CSV 文件为 DataFrame

    Args:
        file: Streamlit UploadedFile 对象

    Returns:
        DataFrame 或 None（解析失败时）
    """
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"CSV 加载失败: {e}")
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

    Args:
        uploaded_file: Streamlit UploadedFile 对象

    Returns:
        DataFrame 或 None（不支持的文件格式 / 解析失败时）
    """
    # 转小写以忽略大小写差异
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return load_csv(uploaded_file)
    elif filename.endswith((".xls", ".xlsx")):
        return load_excel(uploaded_file)
    elif filename.endswith(".json"):
        return load_json(uploaded_file)
    else:
        st.error(f"不支持的文件格式: {uploaded_file.name}")
        return None


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
