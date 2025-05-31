from fastapi import FastAPI, UploadFile, Form
from audio_utils.separator import separate_audio
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()

# Optional: allow frontend or Postman
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

    selected_stems = "calls interpreter from llm part of the project"
    stems = separate_audio(filepath) #actual separation happens here
    #output_path = mix_stems(stems, selected_stems, file.filename)

    return {"output_file": 't'}
