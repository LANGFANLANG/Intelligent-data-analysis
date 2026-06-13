"""
数据库连接管理
───────────────
负责创建和维护 PostgreSQL 的连接池、数据库会话工厂，
并提供 init_db() 函数用于首次启动时自动建表。
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import DATABASE_URL

# ── 数据库引擎 ──
# pool_size:      连接池大小（5个并发连接）
# pool_recycle:   连接回收时间（1小时），防止长连接断开
# pool_pre_ping:  使用前先 ping 测试连接有效性，避免使用已断开的连接
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    pool_recycle=3600,
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
