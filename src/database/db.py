"""
数据库连接管理
───────────────
负责创建和维护 PostgreSQL 的连接池、数据库会话工厂，
并提供 init_db() 函数用于首次启动时自动建表。
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import DATABASE_URL, DB_POOL_SIZE, DB_POOL_RECYCLE

# ── 数据库引擎 ──
engine = create_engine(
    DATABASE_URL,
    pool_size=DB_POOL_SIZE,
    pool_recycle=DB_POOL_RECYCLE,
    pool_pre_ping=True,
)

# ── 会话工厂 ──
# 每次调用 SessionLocal() 产生一个新的数据库会话，
# 用于 with 语句中自动管理连接生命周期
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """
    初始化数据库表结构

    根据 models.py 中定义的所有 Base 子类，在 PostgreSQL 中自动创建表。
    如果表已存在则跳过（不会覆盖或删除已有数据）。
    应在应用启动时调用一次。
    """
    from src.database.models import Base

    # create_all 会检查表是否存在，只创建缺失的表
    Base.metadata.create_all(engine)


def drop_all():
    """删除所有表（危险操作，仅开发调试用）"""
    from src.database.models import Base
    Base.metadata.drop_all(engine)


if __name__ == "__main__":
    import sys
    from src.logger import get_logger
    _log = get_logger("db")

    if "--drop" in sys.argv:
        confirm = input("确认删除所有表? 输入 yes 继续: ")
        if confirm.lower() == "yes":
            drop_all()
            _log.info("所有表已删除")
    else:
        init_db()
        _log.info("数据库表初始化完成（已存在的表自动跳过）")
