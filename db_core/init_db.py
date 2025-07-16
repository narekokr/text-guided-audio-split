
from sqlmodel import SQLModel
from db_core import models
from db_core.config import engine

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    print("created")

if __name__ == "__main__":
    create_db_and_tables()

"""
Read your model definitions (ORM classes)

Generate tables in your PostgreSQL DB

Create them only if they don't exist (safe to run multiple times)
"""