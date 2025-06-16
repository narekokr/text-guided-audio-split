from pydantic import BaseModel

class ResetRequest(BaseModel):
    session_id: str