import uuid
from typing import List

import torchaudio
from fastapi import FastAPI, HTTPException, Path, Query
from pydub import AudioSegment
from pydub.silence import detect_silence
from audio_utils.separator import separate_audio
from audio_utils.remix import handle_remix, session_last_instructions, session_active_task
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles

from db_core.session import (
    get_user_sessions,
    get_session_and_verify_user, get_messages_with_files_for_session_raw,
)
from db_core.config import get_session
from db_core.models import Message, AppSession


from llm_backend.session_manager import (
    get_history,
    save_message,
    get_file_from_db,
    get_or_create_session, save_file_to_db
)

import openai
from llm_backend.interpreter import classify_prompt, describe_audio_edit, parse_feedback
from models.chat_request import ChatRequest
from api.upload import router as upload_router
from models.reset_request import ResetRequest
import numpy as np

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
def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message
    user_id = request.user_id
    get_or_create_session(session_id, user_id)
    save_message(session_id, "user", user_message)
    result = {}
    summary = ""
    audio_path = get_file_from_db(session_id, file_type="uploaded")
    
    # valid_stems = {"vocals", "drums", "bass", "other"}

    is_remix = False
    is_feedback = False
    if session_active_task.get(session_id) == "remix":
        #feedback_keywords = ["more", "less", "bit", "again", "increase", "decrease"]
        feedback_adjustments = parse_feedback(user_message)
        if feedback_adjustments:  # not empty dict
            is_feedback = True
       # if any(kw in user_message.lower() for kw in feedback_keywords):
       # is_feedback = True

    if is_feedback:
        print(f"[DEBUG] Detected feedback prompt for session {session_id}")
        last_instructions = session_last_instructions.get(session_id)
        if not last_instructions:
            reply = "Sorry, I don't have previous settings to adjust. Please provide a new instruction."
            result = {"reply": reply}
        else:
            updated_instructions = increment_instructions_based_on_feedback(user_message, last_instructions)
            result = handle_remix({"type": "remix", "instructions": updated_instructions}, session_id)
            is_remix = True
    else:
        intent = classify_prompt(user_message)
        print(intent)
        if intent["type"] == "separation":
            selected_stems = intent.get("stems", [])
            print(selected_stems)
            separated = []
            silent_stems = []

            if audio_path and selected_stems:
                outputs = separate_audio(audio_path, selected_stems)
                for stem_name, stem_tensor in outputs.items():
                    # Always try to squeeze redundant dimensions
                    stem_tensor = stem_tensor.squeeze()

                    # Now check result
                    if stem_tensor.ndim == 1:
                        stem_tensor = stem_tensor.unsqueeze(0)
                    elif stem_tensor.ndim != 2:
                        raise ValueError(f"Unsupported tensor shape for audio save: {stem_tensor.shape}")

    
                    uid = uuid.uuid4().hex[:6]
                    base = os.path.splitext(os.path.basename(audio_path))[0]
                    output_name = f"{base}_{stem_name}_{uid}.wav"
                    output_path = f"separated/{output_name}"
                                       
                    torchaudio.save(output_path, stem_tensor, 44100)

                    audio = AudioSegment.from_file(output_path, format="wav")

                    silent_ranges = detect_silence(audio, min_silence_len=1000, silence_thresh=-40)
                    is_fully_silent = sum(end - start for start, end in silent_ranges) >= len(audio)

                    if is_fully_silent:
                        silent_stems.append(stem_name)
                    else:
                        url = f"/downloads/{output_name}"
                        separated.append({"name": stem_name, "file_url": url})

                reply = f"Separated stems: {', '.join(s['name'] for s in separated)}. You can download them now."
                stem_names = [s["name"] for s in separated]
                summary = describe_audio_edit("separation", extracted_stems=stem_names)
                if silent_stems:
                    reply += f"Note: The following stems were detected as silent and not included: {', '.join(silent_stems)}."
            else:
                reply = "No audio file or no valid stems selected"
            result = {"reply": reply, "stems": separated}

        elif intent["type"] == "remix":
            is_remix = True
            instructions = intent.get("instructions")
            if instructions:
                result = handle_remix(intent, session_id)
                session_last_instructions[session_id] = intent["instructions"]
                session_active_task[session_id] = "remix"
                summary = describe_audio_edit("remix", instructions=intent["instructions"])
            else:
                result = {
                    "reply": "I couldn't detect any specific remix adjustments. Try saying something like 'make the vocals louder' or 'add reverb to drums'."
                }
        # Fallback
        else:
            result = {
                "reply": "Sorry, I didn't understand your request. Try asking to extract or remix specific stems."
            }


    message_id = save_message(session_id, "assistant", result["reply"])
    if is_remix:
        remix = result.get("remix")
        if remix:
            save_file_to_db(session_id, file_type="remix", path=remix['file_url'], message_id=message_id)
    else:
        stems = result.get("stems", [])
        for stem in stems:
            save_file_to_db(session_id, file_type="stem", path=stem['file_url'], stem=stem['name'], message_id=message_id)

    history = get_history(session_id)

    final_reply = summary if summary else result['reply']

    return {
        "reply": final_reply,
        "stems": result.get("stems", []),
        "remix": result.get("remix"),
        "history": history
    }

@app.post("/reset")
def reset_session_endpoint(request: ResetRequest):
    with get_session() as db:
        session_to_delete = get_session_and_verify_user(db, request.session_id, request.user_id)
        if not session_to_delete:
            raise HTTPException(status_code=404, detail="Session not found or does not belong to user.")

        db.delete(session_to_delete)
        db.commit()

    return {"message": f"Session {request.session_id} for user {request.user_id} has been cleared."}


@app.get("/user/{user_id}/sessions", response_model=List[AppSession])
async def get_user_chat_sessions(user_id: str = Path(...)):
    with get_session() as db:
        sessions = get_user_sessions(db, user_id)
        return sessions


def increment_instructions_based_on_feedback(feedback_text: str, last_instructions: dict) -> dict:
    """
    Adjusts the last instructions for any stems based on user feedback text.
    Works for dynamic stem names (not fixed). Handles 'volumes', 'reverb', 'pitch_shift', 'compression'.
    """
    updated = {k: v.copy() if isinstance(v, dict) else v for k, v in last_instructions.items()}

    # Lowercase feedback for easier matching
    feedback = feedback_text.lower()
    
    # --- VOLUME ---
    if "volume" in feedback or "louder" in feedback or "softer" in feedback or "quiet" in feedback:
        for stem in updated.get("volumes", {}):
            if stem.lower() in feedback:
                if "louder" in feedback or "increase" in feedback or "more" in feedback:
                    updated["volumes"][stem] = min(updated["volumes"].get(stem, 1.0) + 0.1, 2.0)
                elif "softer" in feedback or "decrease" in feedback or "less" in feedback or "quieter" in feedback:
                    updated["volumes"][stem] = max(updated["volumes"].get(stem, 1.0) - 0.1, 0.0)
    
    # --- REVERB ---
    if "reverb" in feedback:
        for stem in updated.get("reverb", {}):
            if stem.lower() in feedback:
                if "more" in feedback or "increase" in feedback:
                    updated["reverb"][stem] = min(updated["reverb"].get(stem, 0.0) + 0.1, 1.0)
                elif "less" in feedback or "decrease" in feedback:
                    updated["reverb"][stem] = max(updated["reverb"].get(stem, 0.0) - 0.1, 0.0)

    # --- PITCH SHIFT ---
    if "pitch" in feedback:
        for stem in updated.get("pitch_shift", {}):
            if stem.lower() in feedback:
                if "up" in feedback or "higher" in feedback or "increase" in feedback:
                    updated["pitch_shift"][stem] += 1
                elif "down" in feedback or "lower" in feedback or "decrease" in feedback:
                    updated["pitch_shift"][stem] -= 1

    # --- COMPRESSION ---
    compression_order = ["low", "medium", "high"]
    if "compression" in feedback:
        for stem in updated.get("compression", {}):
            if stem.lower() in feedback:
                current = updated["compression"].get(stem, "medium")
                idx = compression_order.index(current)
                if "more" in feedback or "increase" in feedback or "stronger" in feedback:
                    idx = min(idx + 1, len(compression_order) - 1)
                elif "less" in feedback or "decrease" in feedback or "softer" in feedback:
                    idx = max(idx - 1, 0)
                updated["compression"][stem] = compression_order[idx]

    return updated


def validate_eq_and_filter(intent):
    for stem, eq in intent.get("instructions", {}).get("eq", {}).items():
        if not all(k in eq for k in ("frequency", "width", "gain_db")):
            raise ValueError(f"Incomplete EQ params for stem {stem}")
    for stem, f in intent.get("instructions", {}).get("filter", {}).items():
        if "type" not in f:
            raise ValueError(f"Missing filter type for stem {stem}")
        if f["type"] in ("lowpass", "highpass") and "cutoff" not in f:
            raise ValueError(f"Missing cutoff for {f['type']} filter on {stem}")
        if f["type"] == "bandpass" and not all(k in f for k in ("low_cutoff", "high_cutoff")):
            raise ValueError(f"Incomplete bandpass filter for {stem}")


@app.get("/session/{session_id}/history")
async def get_session_message_history(
        session_id: str = Path(...),
        user_id: str = Query(..., description="User ID to verify session ownership")
):
    with get_session() as db:
        session = get_session_and_verify_user(db, session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or does not belong to user.")

        messages = get_messages_with_files_for_session_raw(db, session.id)
        audio_path = '/downloads/' + get_file_from_db(session_id, file_type="uploaded")[10:]

        return {"messages": messages, "audio_path": audio_path}
