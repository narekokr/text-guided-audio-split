#response_util

from typing import Dict, Any, Optional, List

def build_chat_response(reply: str, session_id: str,
                       stems: Optional[List] = None,
                       remix: Optional[Dict] = None) -> Dict[str, Any]:
    from llm_backend.session_manager import get_history

    response = {
        "reply": reply,
        "history": get_history(session_id)
    }
    if stems:
        response["stems"] = stems
    if remix:
        response["remix"] = remix
    return response