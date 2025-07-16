from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: str
