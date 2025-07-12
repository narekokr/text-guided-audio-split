from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="appsession.id")
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session: Optional["AppSession"] = Relationship(back_populates="messages")

class File(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="appsession.id")
    file_type: str
    stem: Optional[str]
    path: str
    message_id: Optional[int]
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    session: Optional["AppSession"] = Relationship(back_populates="files")

class AppSession(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = Relationship(back_populates="session")
    files: List[File] = Relationship(back_populates="session")


"""


from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
import datetime

Base = declarative_base() #SQLAlchemy turns Python classes into db_core tables.
#a factory function that returns a base class. you use this base class as the parent for all your model classes
#The Declarative ORM style lets you:

class AppSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.now)
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete"
    )

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    session = relationship("AppSession", back_populates="messages")

class File(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    file_type = Column(String, nullable=False)
    stem = Column(String, nullable=True)
    path = Column(String, nullable=False)  #path to the .wav
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("AppSession", back_populates="files")

"""