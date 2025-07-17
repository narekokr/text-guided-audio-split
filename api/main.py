from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
from typing import Dict
import uvicorn
import openai
from dotenv import load_dotenv
from api.helpers.constants import IntentType, SESSION_TASK_REMIX, DEFAULT_VOLUMES
from api.helpers.request_handlers import (
    handle_separation_request,
    handle_remix_request,
    handle_clarification_request,
    handle_feedback_request
)
from api.helpers.session_state import session_active_task
from api.helpers.validators import validate_chat_request
from api.helpers.response_builders import build_chat_response
from api.helpers.session_state import session_last_instructions
from api.upload import router as upload_router
from models.chat_request import ChatRequest
from models.reset_request import ResetRequest

from llm_backend.interpreter import classify_prompt, parse_feedback
from llm_backend.session_manager import (
    get_or_create_session, save_message, get_history,
    get_session, reset_session
)
from db_core.session import get_user_sessions, get_session_and_verify_user

load_dotenv()
api_key = os.getenv("API_KEY")
client = openai.OpenAI(api_key=api_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)
app.include_router(upload_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/downloads", StaticFiles(directory="separated"), name="downloads")


@app.post("/chat")
async def chat(request: ChatRequest):
    """Process user chat messages for audio separation and remixing."""
    try:
        #validate_chat_request(request)
        
        user_message = request.message
        session_id = request.session_id
        user_id = request.user_id
        
        logger.info(f"Processing chat request from user {user_id}, session {session_id}")
        
        has_remix_output = False
        is_feedback_request = False
        
        intent = classify_prompt(user_message)
        logger.debug(f"Intent classified: {intent}")
        
        if (session_active_task.get(session_id) == SESSION_TASK_REMIX and
            intent["type"] not in [IntentType.SEPARATION.value, IntentType.CLARIFICATION.value]):
            
            feedback_adjustments = parse_feedback(user_message)
            if feedback_adjustments:
                is_feedback_request = True
        
        if is_feedback_request:
            last_instructions = session_last_instructions.get(session_id, {"volumes": DEFAULT_VOLUMES})
            result = handle_feedback_request(user_message, session_id, last_instructions)
            has_remix_output = "remix" in result
        else:
            if intent["type"] == IntentType.SEPARATION.value:
                result = handle_separation_request(intent, session_id)
            elif intent["type"] == IntentType.REMIX.value:
                result = handle_remix_request(intent, session_id)
            elif intent["type"] == IntentType.CLARIFICATION.value:
                result = handle_clarification_request(intent, user_message, session_id)
            else:
                result = {"reply": "I'm not sure how to help with that. Could you try rephrasing your request?"}
            
            has_remix_output = "remix" in result
        
        save_message(session_id, user_id, user_message)
        
        return build_chat_response(
            result["reply"], 
            session_id,
            result.get("stems"),
            result.get("remix")
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        return build_chat_response(
            "Sorry, I encountered an error processing your request. Please try again.",
            request.session_id if hasattr(request, 'session_id') else ""
        )


@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset a user session."""
    try:
        session_exists = get_session_and_verify_user(request.session_id, request.user_id)
        if not session_exists:
            raise HTTPException(status_code=404, detail="Session not found")
        
        reset_session(request.session_id)
        
        session_active_task.pop(request.session_id, None)
        session_last_instructions.pop(request.session_id, None)
        
        logger.info(f"Reset session {request.session_id} for user {request.user_id}")
        return {"message": "Session reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset session")


@app.get("/user/{user_id}/sessions")
async def get_sessions(user_id: str):
    """Get all sessions for a user."""
    try:
        sessions = get_user_sessions(user_id)
        return sessions
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str, user_id: str):
    """Get history for a specific session."""
    try:
        session_exists = get_session_and_verify_user(session_id, user_id)
        if not session_exists:
            raise HTTPException(status_code=404, detail="Session not found")
        
        history = get_history(session_id)
        return history
        
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session history")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
