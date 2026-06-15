"""
数据库 ORM 模型定义
───────────────────
使用 SQLAlchemy 2.0 声明式映射，定义 PostgreSQL 中的两张核心表：

  sessions  ── 会话元数据（名称、关联数据文件、模型参数、时间戳）
  messages  ── 聊天消息（关联到会话，级联删除）

追踪功能已迁移到 Langfuse 观测平台，不再使用本地 traces 表。

关系：Session 1 ── N Message（含外键 + 级联删除）
"""
from sqlalchemy import Column, String, Text, Float, DateTime, Integer, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
import uuid
from datetime import datetime, timezone

# ── 声明基类 ──
Base = declarative_base()


class Session(Base):
    """
    会话表 ── 每个独立的数据分析对话

    字段说明：
      id          - UUID 主键，自动生成全局唯一标识
      name        - 用户可编辑的会话名称
      df_name     - 关联的数据文件名（可选，用于恢复上下文）
      temperature - 模型创造性参数，默认 0.1（偏确定性分析）
      created_at  - 创建时间（UTC）
      updated_at  - 最后更新时间（每次发消息时刷新）
    """
    __tablename__ = "sessions"

    # 主键：UUID 类型，自动生成
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 会话名称，必填
    name = Column(String(255), nullable=False)
    # 关联的数据文件名称，可为空
    df_name = Column(String(500), nullable=True)
    # 模型温度参数
    temperature = Column(Float, default=0.1)
    # 创建和更新时间，默认使用 UTC 当前时间
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # 一对多关系：Session 拥有多条 Message
    # cascade="all, delete-orphan" → 删除会话时级联删除所有消息
    messages = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    """
    消息表 ── 会话中的每条聊天消息

    字段说明：
      id         - 自增主键
      session_id - 外键 → sessions.id，级联删除
      role       - 角色：'user' 或 'assistant'（受 CHECK 约束限制）
      content    - 消息文本内容
      created_at - 消息创建时间（UTC）
    """
    __tablename__ = "messages"

    # 自增主键
    id = Column(Integer, primary_key=True, autoincrement=True)
    # 外键关联到会话表，删除会话时级联删除消息
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    # 消息角色，受 CHECK 约束限制只能为 'user' 或 'assistant'
    role = Column(String(20), CheckConstraint("role IN ('user', 'assistant')"), nullable=False)
    # 消息文本内容
    content = Column(Text, nullable=False)
    # 关联的图表文件路径（JSON数组字符串），如 '["path1.png","path2.png"]'
    images = Column(Text, nullable=True)
    # 消息创建时间
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # 反向关系：每条消息属于一个会话
    session = relationship("Session", back_populates="messages")
