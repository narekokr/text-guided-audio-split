import base64
import io
import torchaudio
from fastapi import FastAPI, UploadFile, Form
from audio_utils.separator import separate_audio
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pydub import AudioSegment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/separate")
async def separate(file: UploadFile, prompt: str = Form(...)):
    os.makedirs("separated", exist_ok=True)
    filepath = f"separated/{file.filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio = AudioSegment.from_file(filepath)
    filepath = filepath.rsplit(".", 1)[0] + "_converted.wav"
    audio.export(filepath, format="wav")

    stems = separate_audio(filepath, prompt)
    stem_data = []
    for i, stem_tensor in stems.items(): #enumerate(stems):
        if stem_tensor.ndim == 3:
            stem_tensor = stem_tensor[0]
        elif stem_tensor.ndim == 1:
            stem_tensor = stem_tensor.unsqueeze(0)

        buf = io.BytesIO()
        torchaudio.save(buf, stem_tensor, 44100, format="wav")
        buf.seek(0)

        b64_audio = base64.b64encode(buf.read()).decode("utf-8")
        stem_data.append({
            "name": f"stem{i}",
            "audio_base64": b64_audio
        })

    return {"stems": stem_data}
