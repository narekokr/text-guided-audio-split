import os
import shutil
from fastapi import UploadFile
from pydub import AudioSegment


def save_uploaded_file(file: UploadFile, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return filepath

def ensure_wav(filepath: str) -> str:
    ext = os.path.splitext(filepath)[-1].lower()
    if ext != ".wav":
        audio = AudioSegment.from_file(filepath)
        filepath = filepath.rsplit(".", 1)[0] + "_converted.wav"
        audio.export(filepath, format="wav")
    return filepath
