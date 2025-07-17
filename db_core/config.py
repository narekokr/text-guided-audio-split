import os

from sqlmodel import create_engine, Session, SQLModel
from dotenv import load_dotenv
load_dotenv()

engine = create_engine(os.getenv('DB_URL'), echo=True)

SQLModel.metadata.create_all(engine)
# Create a session factory
def get_session():
    return Session(engine)
