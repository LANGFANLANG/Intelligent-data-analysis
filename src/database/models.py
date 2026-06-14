"""
数据库 ORM 模型定义
───────────────────
使用 SQLAlchemy 2.0 声明式映射，定义 PostgreSQL 中的三张核心表：

  sessions  ── 会话元数据（名称、关联数据文件、模型参数、时间戳）
  messages  ── 聊天消息（关联到会话，级联删除）
  traces    ── 全链路追踪 span（关联到会话，级联删除）

关系：Session 1 ── N Message / Trace（含外键 + 级联删除）
"""
from sqlalchemy import Column, String, Text, Float, DateTime, Integer, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSON
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
    # Session 拥有多条 Trace
    traces = relationship(
        "Trace", back_populates="session", cascade="all, delete-orphan",
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


class Trace(Base):
    """
    全链路追踪表 ── 记录每次请求的完整执行调用树

    每个 span 对应一个执行单元（分类/LLM调用/工具执行等），
    同一次请求的所有 span 共享同一个 trace_id，
    通过 parent_id 构建树形结构。

    字段说明：
      id          - Span 唯一标识（UUID）
      session_id  - 关联的会话（外键 → sessions.id，级联删除）
      trace_id    - 全链路 ID（同一次请求的所有 span 共享）
      parent_id   - 父 span ID（构建树形结构，根节点为 NULL）
      name        - Span 名称（classify_intent / llm_call / tool_xxx 等）
      start_time  - 开始时间（UTC）
      end_time    - 结束时间（UTC，运行中为 NULL）
      duration_ms - 耗时（毫秒）
      status      - 状态：running / success / error
      metadata    - 扩展信息（JSON）：{tokens, model, intent, error_msg, ...}
      created_at  - 记录创建时间
    """
    __tablename__ = "traces"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    trace_id = Column(String(36), nullable=False, index=True)
    parent_id = Column(String(36), ForeignKey("traces.id"), nullable=True)
    name = Column(String(128), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="running")
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    session = relationship("Session", back_populates="traces")
