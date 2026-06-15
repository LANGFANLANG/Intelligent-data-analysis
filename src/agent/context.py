"""
工具上下文模块
──────────────
通过 contextvars 为每个请求/线程隔离 DataFrame 访问入口。
工具函数不依赖 Streamlit 的 session_state，而是通过 ToolContext 获取数据，
保持工具层的独立性。

数据源支持:
  - "file": 用户上传的文件数据（原有模式）
  - "database": 数据库直连模式（sql_query 工具产生的结果）

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

_df = contextvars.ContextVar[pd.DataFrame | None]("tool_df", default=None)
_df_name = contextvars.ContextVar[str | None]("tool_df_name", default=None)
_data_source = contextvars.ContextVar[str]("tool_data_source", default="file")


class ToolContext:

    @classmethod
    def set(cls, df: pd.DataFrame, name: str):
        _df.set(df)
        _df_name.set(name)

    @classmethod
    def get(cls) -> pd.DataFrame | None:
        return _df.get()

    @classmethod
    def get_name(cls) -> str | None:
        return _df_name.get()

    @classmethod
    def clear(cls):
        _df.set(None)
        _df_name.set(None)

    @classmethod
    def has_data(cls) -> bool:
        return _df.get() is not None

    @classmethod
    def set_data_source(cls, source: str):
        _data_source.set(source)

    @classmethod
    def get_data_source(cls) -> str:
        return _data_source.get()

    @classmethod
    def is_database_mode(cls) -> bool:
        return _data_source.get() == "database"

    @classmethod
    def set_query_result(cls, df: pd.DataFrame, label: str = "sql_query_result"):
        _df.set(df)
        _df_name.set(label)
