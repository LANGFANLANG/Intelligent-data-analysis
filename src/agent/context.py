"""
工具上下文模块
──────────────
通过单例模式为所有 LangChain Tool 提供统一的 DataFrame 访问入口。
工具函数不依赖 Streamlit 的 session_state，而是通过 ToolContext 获取数据，
保持工具层的独立性。

用法:
    from src.agent.context import ToolContext
    ToolContext.set(df, "sales.csv")
    # 工具函数内部:
    df = ToolContext.get()
"""
import pandas as pd


class ToolContext:
    """
    工具上下文单例

    持有当前分析会话的 DataFrame 和文件名，
    供所有工具函数读取数据。
    """
    _df: pd.DataFrame | None = None
    _df_name: str | None = None

    @classmethod
    def set(cls, df: pd.DataFrame, name: str):
        """
        设置当前数据上下文

        Args:
            df:   已加载的 DataFrame
            name: 文件名标识
        """
        cls._df = df
        cls._df_name = name

    @classmethod
    def get(cls) -> pd.DataFrame | None:
        """获取当前 DataFrame"""
        return cls._df

    @classmethod
    def get_name(cls) -> str | None:
        """获取当前数据文件名"""
        return cls._df_name

    @classmethod
    def clear(cls):
        """清空上下文"""
        cls._df = None
        cls._df_name = None

    @classmethod
    def has_data(cls) -> bool:
        """检查是否有数据加载"""
        return cls._df is not None
