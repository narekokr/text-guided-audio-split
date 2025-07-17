from models.chat_request import ChatRequest

def validate_chat_request(request: ChatRequest) -> None:
    if not request.message.strip():
        raise ValueError("Message cannot be empty")
    if not request.session_id:
        raise ValueError("Session ID is required")
    if not request.user_id:
        raise ValueError("User ID is required")