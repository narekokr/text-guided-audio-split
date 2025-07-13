from sqlalchemy import text
from sqlmodel import Session, select
from typing import List, Optional

from db_core.models import AppSession, Message, File
from db_core.config import get_session


def ensure_session_exists(session: Session, session_id: str, user_id: str):
    existing = session.exec(select(AppSession).where(AppSession.id == session_id)).first()
    if not existing:
        session.add(AppSession(id=session_id, user_id=user_id))
        session.commit()


def get_user_sessions(session: Session, user_id: str) -> List[AppSession]:
    sessions = session.exec(
        select(AppSession)
        .where(AppSession.user_id == user_id)
        .order_by(AppSession.created_at.desc())
    ).all()
    return sessions


def get_session_and_verify_user(session: Session, session_id: str, user_id: str) -> Optional[AppSession]:
    app_session = session.exec(
        select(AppSession)
        .where(AppSession.id == session_id, AppSession.user_id == user_id)
    ).first()
    return app_session


def get_messages_with_files_for_session_raw(db_connection, session_id: str):
    messages = {}

    sql = """
          SELECT m.id AS message_id,
                 m.content,
                 m.timestamp,
                 m.role,
                 f.id AS file_id,
                 f.file_type,
                 f.stem,
                 f.path
          FROM message AS m
                   LEFT JOIN file AS f ON m.id = f.message_id
          WHERE m.session_id = :sid
          ORDER BY m.timestamp, m.id; \
          """

    results = db_connection.exec(text(sql), params={"sid": session_id})

    for row in results.all():
        message_id, content, timestamp, role, file_id, file_type, stem, path = row

        if message_id not in messages:
            messages[message_id] = {
                "id": message_id,
                "content": content,
                "timestamp": timestamp,
                "role": role,
                "files": []
            }

        if file_id is not None:
            messages[message_id]["files"].append({
                "id": file_id,
                "file_type": file_type,
                "stem": stem,
                "role": role,
                "file_url": path
            })

    return list(messages.values())