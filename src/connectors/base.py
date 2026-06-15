"""
数据库连接器抽象基类
───────────────────
定义数据库连接器的统一接口，所有具体数据库实现必须继承此类。

接口方法:
  connect()            - 建立连接
  execute_query()      - 执行只读 SQL 查询，返回 DataFrame
  discover_tables()    - 获取所有表名
  describe_table()     - 获取单表结构（列名/类型/可空/键）
  get_table_row_count()- 获取表大致行数
  test_connection()    - 连接测试
  close()              - 关闭连接池
  is_connected()       - 连接状态

设计原则:
  - 所有方法为 @abstractmethod，子类必须实现
  - 查询结果统一返回 pandas DataFrame，方便后续分析工具链复用
  - 不假设具体数据库类型，仅定义行为契约
"""
from abc import ABC, abstractmethod
import pandas as pd


class AbstractConnector(ABC):
    """数据库连接器抽象基类

    职责:
      封装数据库连接、查询执行和元数据发现。
      所有具体数据库实现（MySQL/PostgreSQL/SQLite 等）继承此类。
    """

    @abstractmethod
    def connect(self) -> bool:
        """建立数据库连接

        Returns:
            True 表示连接成功，False 表示失败
        """
        ...

    @abstractmethod
    def execute_query(self, sql: str, max_rows: int = 1000, timeout: int = 30) -> pd.DataFrame:
        """执行只读 SQL 查询

        Args:
            sql:      SQL 查询语句（仅允许 SELECT）
            max_rows: 最大返回行数，超过则自动截断
            timeout:  查询超时秒数

        Returns:
            查询结果 DataFrame

        Raises:
            RuntimeError: 数据库未连接
            ValueError:   非 SELECT 语句
        """
        ...

    @abstractmethod
    def discover_tables(self) -> list[str]:
        """获取数据库中所有用户表名

        Returns:
            表名列表，按字母序排列
        """
        ...

    @abstractmethod
    def describe_table(self, table_name: str) -> list[dict]:
        """获取单表的完整结构信息

        Args:
            table_name: 表名

        Returns:
            列信息列表，每项包含: name, type, nullable, default, comment, is_primary_key
        """
        ...

    @abstractmethod
    def get_table_row_count(self, table_name: str) -> int | None:
        """获取表的大致行数

        Args:
            table_name: 表名

        Returns:
            行数估算值；获取失败时返回 None
        """
        ...

    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """测试当前连接是否可用

        Returns:
            (是否成功, 失败原因)
        """
        ...

    @abstractmethod
    def close(self):
        """关闭连接池，释放所有数据库资源"""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """连接状态: True 表示已建立且可用"""
        ...
