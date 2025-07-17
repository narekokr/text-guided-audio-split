import uuid
from typing import List
import torchaudio
from fastapi import FastAPI, HTTPException, Path, Query
from pydub import AudioSegment
from pydub.silence import detect_silence
from audio_utils.separator import separate_audio
from audio_utils.remix import (
    handle_remix,
    session_last_instructions,
    session_active_task
)
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles
from db_core.session import (
    get_user_sessions,
    get_session_and_verify_user,
    get_messages_with_files_for_session_raw,
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
from llm_backend.interpreter import classify_prompt, describe_audio_edit, parse_feedback, describe_feedback_changes, generate_clarification_response, apply_feedback_to_instructions
from models.chat_request import ChatRequest
from api.upload import router as upload_router
from models.reset_request import ResetRequest
import numpy as np

app = FastAPI(debug=True)

with open("api_key.txt", "r") as file:
    api_key = file.read().strip()

client = openai.OpenAI(api_key=api_key)
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
    valid_stems = {"vocals", "drums", "bass", "other"}

    is_remix = False
    is_feedback = False

    # First, always classify the intent to see what the user really wants
    intent = classify_prompt(user_message)

    # Only try feedback parsing if it's not a clear separation or clarification request
    if session_active_task.get(session_id) == "remix" and intent["type"] not in ["separation", "clarification"]:
        feedback_adjustments = parse_feedback(user_message)
        if feedback_adjustments:
            is_feedback = True

    if is_feedback:
        last_instructions = session_last_instructions.get(session_id)
        if not last_instructions:
            reply = "Sorry, I don't have previous settings to adjust. Please provide a new instruction."
            result = {"reply": reply}
        else:
            updated_instructions = apply_feedback_to_instructions(feedback_adjustments, last_instructions)

            if updated_instructions == last_instructions:
                result = {
                    "reply": "I couldn't detect any changes to make based on your request. Could you be more specific about what you'd like me to adjust?"
                }
            else:
                result = handle_remix({"type": "remix", "instructions": updated_instructions}, session_id)
                incremental_summary = describe_feedback_changes(user_message, last_instructions, updated_instructions)
                result["reply"] = incremental_summary
                is_remix = True
    else:
        # Use the already classified intent

        if intent["type"] == "separation":
            selected_stems = intent.get("stems", [])
            if not selected_stems:
                selected_stems = list(valid_stems)

            separated = []
            silent_stems = []
            invalid_stems = [s for s in selected_stems if s not in valid_stems]
            selected_stems = [s for s in selected_stems if s in valid_stems]

            if invalid_stems:
                reply = f"Note: The following stems are not supported and will be ignored: {', '.join(invalid_stems)}.\n"

            if audio_path and selected_stems:
                outputs = separate_audio(audio_path, selected_stems)
                for stem_name, stem_tensor in outputs.items():
                    if stem_tensor.ndim == 3:
                        stem_tensor = stem_tensor[0]
                    elif stem_tensor.ndim == 1:
                        stem_tensor = stem_tensor.unsqueeze(0)

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
                result["reply"] = summary
            else:
                result = {
                    "reply": "I couldn't detect any specific remix adjustments. Try saying something like 'make the vocals louder' or 'add reverb to drums'."
                }

        elif intent["type"] == "clarification":
            reason = intent.get("reason", "unclear_intent")
            has_audio = audio_path is not None
            clarification_reply = generate_clarification_response(reason, user_message, has_audio)
            result = {"reply": clarification_reply}

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

    return {
        "reply": result['reply'],
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
    Helper function to adjust instructions dynamically
    Works with general phrases like “make vocals louder” “more reverb on drums” “pitch vocals up”
    Uses last applied instructions as baseline
    Returns updated instructions.

    Volumes: Adjust ±0.1 within [0.0, 2.0] based on feedback.

    Reverb: Adjust ±0.1 within [0.0, 1.0].

    Pitch shift: Adjust ±1 semitone per feedback.

    Compression: Switch between "low", "medium", "high" based on feedback
    """

    updated = last_instructions.copy()
    volume_keywords = {
        "vocals": "vocals",
        "drums": "drums",
        "bass": "bass",
        "other": "other"
    }
    if "volume" in feedback_text or "louder" in feedback_text or "softer" in feedback_text:
        for stem, stem_key in volume_keywords.items():
            if stem in feedback_text:
                if "louder" in feedback_text or "increase" in feedback_text or "more" in feedback_text:
                    updated["volumes"][stem_key] = min(updated["volumes"].get(stem_key, 1.0) + 0.1, 2.0)
                elif "softer" in feedback_text or "decrease" in feedback_text or "less" in feedback_text:
                    updated["volumes"][stem_key] = max(updated["volumes"].get(stem_key, 1.0) - 0.1, 0.0)

    if "reverb" in feedback_text:
        for stem in updated["reverb"]:
            if "more" in feedback_text or "increase" in feedback_text:
                updated["reverb"][stem] = min(updated["reverb"].get(stem, 0.0) + 0.1, 1.0)
            elif "less" in feedback_text or "decrease" in feedback_text:
                updated["reverb"][stem] = max(updated["reverb"].get(stem, 0.0) - 0.1, 0.0)

    if "pitch" in feedback_text:
        for stem in updated["pitch_shift"]:
            if "up" in feedback_text or "higher" in feedback_text or "increase" in feedback_text:
                updated["pitch_shift"][stem] += 1
            elif "down" in feedback_text or "lower" in feedback_text or "decrease" in feedback_text:
                updated["pitch_shift"][stem] -= 1

    compression_order = ["low", "medium", "high"]
    if "compression" in feedback_text:
        for stem in updated["compression"]:
            current = updated["compression"].get(stem, "medium")
            idx = compression_order.index(current)
            if "more" in feedback_text or "increase" in feedback_text or "stronger" in feedback_text:
                idx = min(idx + 1, len(compression_order) - 1)
            elif "less" in feedback_text or "decrease" in feedback_text or "softer" in feedback_text:
                idx = max(idx - 1, 0)
            updated["compression"][stem] = compression_order[idx]

    return updated

def increment_instructions_based_on_feedback_phase2(feedback_adjustments: dict, last_instructions: dict) -> dict:
    updated = last_instructions.copy()

    volume_map = {
        "slightly softer": -0.1,
        "softer": -0.3,
        "much softer": -0.6,
        "mute": -1.0,
        "slightly louder": +0.1,
        "louder": +0.3,
        "much louder": +0.6
    }
    for stem, change in feedback_adjustments.get("volumes", {}).items():
        delta = volume_map.get(change, 0.0)
        if delta != 0.0:
            updated["volumes"][stem] = np.clip(updated["volumes"].get(stem, 1.0) + delta, 0.0, 2.0)

    for stem, change in feedback_adjustments.get("reverb", {}).items():
        if change == "more":
            updated["reverb"][stem] = min(updated["reverb"].get(stem, 0.0) + 0.1, 1.0)
        elif change == "less":
            updated["reverb"][stem] = max(updated["reverb"].get(stem, 0.0) - 0.1, 0.0)

    for stem, change in feedback_adjustments.get("pitch_shift", {}).items():
        if change.startswith("+") or change.startswith("-"):
            updated["pitch_shift"][stem] += int(change)

    compression_order = ["low", "medium", "high"]
    for stem, level in feedback_adjustments.get("compression", {}).items():
        updated["compression"][stem] = level if level in compression_order else updated["compression"].get(stem, "medium")

    return updated

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