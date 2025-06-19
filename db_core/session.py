from sqlmodel import Session
from sqlalchemy import select

from db_core.models import AppSession  # adjust the import to your model path
from db_core.config import get_session

def ensure_session_exists(session: Session, session_id: str):
    existing = session.exec(select(AppSession).where(AppSession.id == session_id)).first()
    if not existing:
        session.add(AppSession(id=session_id))
        session.commit()
