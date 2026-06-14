"""
工具上下文模块
──────────────
通过 contextvars 为每个请求/线程隔离 DataFrame 访问入口。
工具函数不依赖 Streamlit 的 session_state，而是通过 ToolContext 获取数据，
保持工具层的独立性。

设计要点:
  - contextvars.ContextVar 保证多用户/多线程并发时数据不串扰
  - 每个 Streamlit session 在独立的脚本运行中 set/get 自己的数据
  - set() 在 UI 层（用户上传数据时调用），get() 在工具层（Agent 调用工具时读取）

用法:
    from src.agent.context import ToolContext
    ToolContext.set(df, "sales.csv")
    # 工具函数内部:
    df = ToolContext.get()
"""
import pandas as pd
import contextvars

# ── 每个线程/协程独立的上下文变量 ──
_df = contextvars.ContextVar[pd.DataFrame | None]("tool_df", default=None)
_df_name = contextvars.ContextVar[str | None]("tool_df_name", default=None)


class ToolContext:
    """
    工具上下文

    持有当前分析会话的 DataFrame 和文件名，
    供所有工具函数读取数据。
    基于 contextvars 实现线程/协程隔离，
    多用户并发时互不干扰。
    """

    @classmethod
    def set(cls, df: pd.DataFrame, name: str):
        """
        设置当前数据上下文

        Args:
            df:   已加载的 DataFrame
            name: 文件名标识
        """
        _df.set(df)
        _df_name.set(name)

    @classmethod
    def get(cls) -> pd.DataFrame | None:
        """获取当前 DataFrame"""
        return _df.get()

    @classmethod
    def get_name(cls) -> str | None:
        """获取当前数据文件名"""
        return _df_name.get()

    @classmethod
    def clear(cls):
        """清空上下文"""
        _df.set(None)
        _df_name.set(None)

    @classmethod
    def has_data(cls) -> bool:
        """检查是否有数据加载"""
        return _df.get() is not None
