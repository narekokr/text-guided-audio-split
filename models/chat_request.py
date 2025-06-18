from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str

# Pydantic models — they represent API input, not DB storage.

