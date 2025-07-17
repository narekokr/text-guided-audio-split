from fastapi import UploadFile, Form, APIRouter
from pydub import AudioSegment
import os, uuid, shutil
from llm_backend.session_manager import save_file_to_db
from db_core.session import ensure_session_exists
from db_core.config import get_session
router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile, session_id: str = Form(...), user_id: str = Form(...)):
    output_dir = "separated"
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join(output_dir, file.filename)
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio = AudioSegment.from_file(original_path)
    converted_path = original_path.rsplit(".", 1)[0] + "_converted.wav"
    audio.export(converted_path, format="wav")

    with get_session() as db:
        ensure_session_exists(db, session_id, user_id)

    save_file_to_db(session_id, file_type="uploaded",  path=converted_path, stem=None)
    print(f"stored file path: {converted_path} for session: {session_id} by user: {user_id}")

    return {
        "message": "File uploaded and converted to WAV",
        "session_id": session_id,
        "user_id": user_id,
        "converted_path": converted_path
    }