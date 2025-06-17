from fastapi import UploadFile, Form, APIRouter
from pydub import AudioSegment
import os, uuid, shutil
from llm_backend.chat_manager import session_manager

#audio file logic
router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile, session_id: str = Form(...)):
    output_dir = "separated"
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join(output_dir, file.filename)
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert to WAV
    audio = AudioSegment.from_file(original_path)
    converted_path = original_path.rsplit(".", 1)[0] + "_converted.wav"
    audio.export(converted_path, format="wav")

    session_manager.set_file(session_id, converted_path)
    print(f"stored file path: {converted_path} for session: {session_id}")

    return {
        "message": "File uploaded and converted to WAV",
        "session_id": session_id,
        "converted_path": converted_path
    }
