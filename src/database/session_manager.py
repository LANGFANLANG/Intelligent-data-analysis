"""
会话管理业务层
──────────────
提供会话的完整 CRUD 操作，所有方法封装了对 PostgreSQL 的读写。

每个方法使用 with SessionLocal() 创建独立的数据库会话，
确保连接及时释放，避免连接泄漏。

API 清单：
  create_session()  - 创建新会话
  get_session()     - 读取单个会话（含全部消息）
  list_sessions()   - 列出所有会话（按更新时间倒序）
  add_message()     - 追加消息并刷新会话的 updated_at
  delete_session()  - 删除会话（级联删除所有消息）
  rename_session()  - 重命名会话
  update_meta()     - 更新会话元数据（df_name / temperature）
"""
from datetime import datetime, timezone
import json
from src.database.db import SessionLocal
from src.database.models import Session, Message


def _parse_images(raw: str | None) -> list[str]:
    """解析数据库中存储的图表路径 JSON数组"""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


class SessionManager:
    """会话管理器 ── 提供会话生命周期的全部操作"""

    @staticmethod
    def create_session(name: str | None = None, df_name: str | None = None) -> str:
        """
        创建一个新的分析会话

        Args:
            name:    会话名称，为空时自动生成 "新会话 MM-DD HH:MM"
            df_name: 关联的数据文件名（可选）

        Returns:
            新会话的 UUID 字符串
        """
        with SessionLocal() as db:
            now = datetime.now(timezone.utc)
            if not name:
                name = f"新会话 {now.strftime('%m-%d %H:%M')}"
            session = Session(name=name, df_name=df_name, created_at=now, updated_at=now)
            db.add(session)
            db.commit()
            return str(session.id)

    @staticmethod
    def get_session(session_id: str) -> dict | None:
        """
        读取单个会话的完整数据（含全部历史消息）

        Args:
            session_id: 会话 UUID

        Returns:
            包含会话元数据和消息列表的字典，不存在时返回 None
            格式: {id, name, df_name, temperature, created_at, updated_at, messages: [{role, content}, ...]}
        """
        with SessionLocal() as db:
            # 查询会话及其关联的全部消息（通过 relationship 自动加载）
            session = db.query(Session).filter(Session.id == session_id).first()
            if not session:
                return None
            return {
                "id": str(session.id),
                "name": session.name,
                "df_name": session.df_name,
                "temperature": session.temperature or 0.1,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "messages": [
                    {"role": m.role, "content": m.content, "images": _parse_images(m.images)}
                    for m in session.messages
                ],
            }

    @staticmethod
    def list_sessions() -> list[dict]:
        """
        列出全部会话，按更新时间倒序排列

        Returns:
            会话摘要列表，每个元素包含: {id, name, df_name, created_at, updated_at, message_count}
        """
        with SessionLocal() as db:
            sessions = db.query(Session).order_by(Session.updated_at.desc()).all()
            return [
                {
                    "id": str(s.id),
                    "name": s.name,
                    "df_name": s.df_name,
                    "created_at": s.created_at.strftime("%m-%d %H:%M"),
                    "updated_at": s.updated_at.strftime("%m-%d %H:%M"),
                    "message_count": len(s.messages),
                }
                for s in sessions
            ]

    @staticmethod
    def add_message(session_id: str, role: str, content: str, images: list[str] | None = None):
        """
        向指定会话追加一条消息，并同步刷新会话的 updated_at 时间

        Args:
            session_id: 目标会话 UUID
            role:       消息角色 ('user' / 'assistant')
            content:    消息文本内容
            images:     关联的图表文件路径列表（可选）
        """
        with SessionLocal() as db:
            # 创建消息记录
            images_json = json.dumps(images) if images else None
            msg = Message(session_id=session_id, role=role, content=content, images=images_json)
            db.add(msg)
            # 同步更新会话的最后活跃时间
            db.query(Session).filter(Session.id == session_id).update(
                {"updated_at": datetime.now(timezone.utc)}
            )
            db.commit()

    @staticmethod
    def delete_session(session_id: str):
        """
        删除指定会话及其全部关联消息

        通过数据库外键的 ON DELETE CASCADE 约束，
        删除 sessions 记录时会自动删除对应的所有 messages 记录。

        Args:
            session_id: 待删除的会话 UUID
        """
        with SessionLocal() as db:
            db.query(Session).filter(Session.id == session_id).delete()
            db.commit()

    @staticmethod
    def rename_session(session_id: str, name: str):
        """
        重命名指定会话

        Args:
            session_id: 目标会话 UUID
            name:       新的会话名称
        """
        with SessionLocal() as db:
            db.query(Session).filter(Session.id == session_id).update(
                {"name": name, "updated_at": datetime.now(timezone.utc)}
            )
            db.commit()

    @staticmethod
    def update_meta(session_id: str, **kwargs):
        """
        更新会话的元数据字段

        仅允许更新白名单中的字段（df_name / temperature），
        防止误修改 id 或时间戳等受保护字段。

        Args:
            session_id:   目标会话 UUID
            **kwargs:     需更新的字段键值对 (df_name=..., temperature=...)
        """
        allowed = {"df_name", "temperature", "created_at"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if updates:
            updates["updated_at"] = datetime.now(timezone.utc)
            with SessionLocal() as db:
                db.query(Session).filter(Session.id == session_id).update(updates)
                db.commit()
