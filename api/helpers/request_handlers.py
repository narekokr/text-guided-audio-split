import logging
import os
from typing import Dict, Any
from pydub import AudioSegment
from pydub.silence import detect_silence
from audio_utils.remix import handle_remix
from audio_utils.separator import separate_audio
from llm_backend.interpreter import parse_feedback, apply_feedback_to_instructions, describe_feedback_changes, \
    describe_audio_edit, generate_clarification_response
from llm_backend.session_manager import get_file_from_db
from api.helpers.constants import (
    SESSION_TASK_SEPARATION, SESSION_TASK_REMIX,
)
import uuid
import torchaudio
from api.helpers.session_state import session_active_task, session_last_instructions

logger = logging.getLogger(__name__)

def handle_feedback_request(user_message: str, session_id: str,
                             last_instructions: Dict) -> Dict[str, Any]:
    """Handle feedback on existing remix."""
    logger.info(f"Processing feedback request for session {session_id}")

    feedback_adjustments = parse_feedback(user_message)
    updated_instructions = apply_feedback_to_instructions(feedback_adjustments, last_instructions)

    if updated_instructions == last_instructions:
        return {
            "reply": "I couldn't detect any changes to make based on your request. Could you be more specific about what you'd like me to adjust?"
        }

    result = handle_remix({"type": "remix", "instructions": updated_instructions}, session_id)

    incremental_summary = describe_feedback_changes(user_message, last_instructions, updated_instructions)
    result["reply"] = incremental_summary

    session_last_instructions[session_id] = updated_instructions

    return result


def handle_separation_request(intent: Dict, session_id: str) -> Dict[str, Any]:
    """Handle audio separation request."""
    logger.info(f"Processing separation request for session {session_id}")

    audio_path = get_file_from_db(session_id)
    if not audio_path:
        return {"reply": "No audio file found for separation."}

    valid_stems = ["vocals", "drums", "bass", "other"]
    selected_stems = intent.get("stems", [])

    if not selected_stems:
        selected_stems = list(valid_stems)

    separated = []
    silent_stems = []
    invalid_stems = [s for s in selected_stems if s not in valid_stems]
    selected_stems = [s for s in selected_stems if s in valid_stems]

    reply = ""
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

    if separated:
        stem_names = [s["name"] for s in separated]
        reply += describe_audio_edit("separation", extracted_stems=stem_names)
    else:
        reply += "No audio content found in the requested stems."

    if silent_stems:
        reply += f" Note: {', '.join(silent_stems)} appear to be silent in this track."

    session_active_task[session_id] = SESSION_TASK_SEPARATION

    return {"reply": reply, "stems": separated}


def handle_remix_request(intent: Dict, session_id: str) -> Dict[str, Any]:
    """Handle audio remix request."""
    logger.info(f"Processing remix request for session {session_id}")

    result = handle_remix(intent, session_id)

    summary = describe_audio_edit("remix", instructions=intent["instructions"])
    result["reply"] = summary

    session_active_task[session_id] = SESSION_TASK_REMIX
    session_last_instructions[session_id] = intent["instructions"]

    return result


def handle_clarification_request(intent: Dict, user_message: str,
                                  session_id: str) -> Dict[str, Any]:
    """Handle clarification request."""
    logger.info(f"Processing clarification request for session {session_id}")

    has_audio = session_id in session_active_task
    clarification_response = generate_clarification_response(
        intent["reason"], user_message, has_audio
    )

    return {"reply": clarification_response}
