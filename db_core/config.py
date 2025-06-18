from sqlmodel import create_engine, Session

# Create the engine (this part you already have)
engine = create_engine("postgresql+psycopg2://lelat@localhost:5432/audio_split", echo=True)

# Create a session factory
def get_session():
    return Session(engine)
