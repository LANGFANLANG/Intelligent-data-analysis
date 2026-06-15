"""
数据库连接器包
──────────────
提供统一的数据源抽象层，支持多种数据库的连接、查询和元数据发现。

当前支持:
  - MySQL (pymysql + SQLAlchemy)

设计要点:
  - AbstractConnector 定义统一接口，方便后续扩展其他数据库
  - MySQLConnector 实现连接池、只读事务、超时控制
  - create_connector() 为全局工厂函数，基于配置自动创建连接

用法:
    from src.connectors import create_connector, get_connector
    connector = create_connector()
    tables = connector.discover_tables()
"""
from src.connectors.base import AbstractConnector
from src.connectors.mysql_connector import MySQLConnector, create_connector, close_connector, get_connector
