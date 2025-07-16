from db_core.models import AppSession, Message, File
from db_core.config import get_session


def get_or_create_session(session_id: str, user_id: str):
    with get_session() as db:
        session = db.get(AppSession, session_id)
        if not session:
            session = AppSession(id=session_id, user_id=user_id)
            db.add(session)
            db.commit()
        return session

def get_history(session_id: str) -> list[dict]:
    with get_session() as db:
        messages = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp)
            .all()
        )
        return [{"role": m.role, "content": m.content} for m in messages]

def save_message(session_id: str, role: str, content: str):
    with get_session() as db:
        msg = Message(session_id=session_id, role=role, content=content)
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg.id

def reset_session(session_id: str):
    with get_session() as db:
        db.query(Message).filter(Message.session_id == session_id).delete()
        db.query(File).filter(File.session_id == session_id).delete()
        db.query(AppSession).filter(AppSession.id == session_id).delete()
        db.commit()

def get_file_from_db(session_id: str, file_type: str = "uploaded"):
    with get_session() as db:
        file = (
            db.query(File)
            .filter(File.session_id == session_id, File.file_type == file_type)
            .order_by(File.uploaded_at.desc())  # get latest
            .first()
        )
        return file.path if file else None

def save_file_to_db(session_id: str, file_type: str, path: str, stem: str = None, message_id: int = None):
    with get_session() as db:
        file = File(session_id=session_id, file_type=file_type, path=path, stem=stem, message_id=message_id)
        db.add(file)
        db.commit()

