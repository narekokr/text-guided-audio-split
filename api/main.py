import uuid
import torchaudio
from fastapi import FastAPI, UploadFile, Form
from audio_utils.separator import separate_audio
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pydub import AudioSegment
from fastapi.staticfiles import StaticFiles
from llm_backend.chat_manager import SessionManager
import openai
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
from models.ChatRequest import ChatRequest

app = FastAPI()
session_manager = SessionManager()
api_key = "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"
client = openai.OpenAI(api_key=api_key)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/downloads", StaticFiles(directory="separated"), name="downloads")

@app.post("/separate")
async def separate(file: UploadFile, prompt: str = Form(...)):
    output_dir = "separated"
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"separated/{file.filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    audio = AudioSegment.from_file(filepath)
    filepath = filepath.rsplit(".", 1)[0] + "_converted.wav"
    audio.export(filepath, format="wav")

    stems = separate_audio(filepath, prompt)

    #output dictionary

    stem_data = []
    for stem_name, stem_tensor in stems.items(): #enumerate(stems):
        if stem_tensor.ndim == 3:
            stem_tensor = stem_tensor[0]
        elif stem_tensor.ndim == 1:
            stem_tensor = stem_tensor.unsqueeze(0)

        unique_id = uuid.uuid4().hex[:6]
        base = os.path.splitext(os.path.basename(filepath))[0]
        output_name = f"{base}_{stem_name}_{unique_id}.wav"
        output_path = f"separated/{output_name}"

        print(f"Saving stem '{stem_name}' with shape {stem_tensor.shape} to {output_path}")

        try:
            torchaudio.save(output_path, stem_tensor, 44100)  # stem_tensor should be 2d
        except Exception as e:
            raise RuntimeError(f"Failed to save {output_path}: {e}")

        stem_data.append({
            "name": stem_name,
            "file_name": f"{output_name}"
        })

    return {"stems": stem_data }

@app.post("/chat")
def chat(request: ChatRequest):
    session_manager.add_message(request.session_id, "user")

    history = session_manager.get_history(request.session_id)

    message_objects = [
        ChatCompletionUserMessageParam(**msg) if msg["role"] == "user"
        else ChatCompletionAssistantMessageParam(**msg)
        for msg in history
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message_objects,
        temperature=0,
    )

    reply = response.choices[0].message.content
    session_manager.add_message(request.session_id, "assistant", reply)

    response = {
        "reply": reply,
        "history": history
    }
    return response


"""
#TODO
@app.post("/chat") - for multi-turn conversational guidance
GET /status — for long jobs or async audio processing or reporting what has been separated or downloaded
GET /stems/{id} — to retrieve previously generated files
/downloads - static files for access to ouput
optional extension: /upload for uploading audio in advance (store in session)
/reset	Clear session history
"""