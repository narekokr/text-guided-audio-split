import uuid
import torchaudio
from fastapi import FastAPI
from audio_utils.separator import separate_audio
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles
from llm_backend.chat_manager import session_manager
import openai
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, \
    ChatCompletionSystemMessageParam

from llm_backend.interpreter import interpret_prompt
from models.ChatRequest import ChatRequest
from api.upload import router as upload_router

app = FastAPI()
api_key = "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"
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
    """
        Main chat endpoint handling conversational flow and orchestration.

        - Logs user message and maintains session history.
        - Uses LLM to interpret user intent.
        - If the message implies a stem separation request and a file is uploaded:
            - Calls separation logic and returns downloadable audio stems.
            - Crafts a helpful assistant message confirming stem extraction.
        - Otherwise, responds with natural chat-style LLM response.
    """
    user_message = request.message
    session_manager.add_message(request.session_id, "user", user_message)
    history = session_manager.get_history(request.session_id)

    messages = [
       ChatCompletionSystemMessageParam(
           role="system",
           content=(
               "You are a music assistant that helps users extract stems from uploaded audio. "
               "Respond naturally. If stems were separated, confirm and provide file names."
           )
       )
   ] + [
       ChatCompletionUserMessageParam(**m) if m["role"] == "user"
       else ChatCompletionAssistantMessageParam(**m)
       for m in history
   ]

    selected_stems = interpret_prompt(request.message)
    audio_path = session_manager.get_file(request.session_id)
    separated = []

    if audio_path and selected_stems:
        outputs = separate_audio(audio_path, selected_stems)
        for stem_name, stem_tensor in outputs.items():
            if stem_tensor.ndim == 3:
                stem_tensor = stem_tensor[0]
            elif stem_tensor.ndim == 1:
                stem_tensor = stem_tensor.unsqueeze(0)

            uid = uuid.uuid4().hex[:6]
            base = os.path.splitext(os.path.basename(audio_path))[0]
            outname = f"{base}_{stem_name}_{uid}.wav"
            outpath = f"separated/{outname}"
            torchaudio.save(outpath, stem_tensor, 44100)
            url = f"/downloads/{outname}"
            separated.append({"name": stem_name, "file_url": url})

        reply = f"✅ Separated stems: {', '.join(s['name'] for s in separated)}. You can download them now."
    else:
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        reply = chat_response.choices[0].message.content

    session_manager.add_message(request.session_id, "assistant", reply)
    response = {
        "reply": reply,
        "stems": separated,
        "history": history
    }
    return response


"""
#TODO
GET /status — for long jobs or async audio processing or reporting what has been separated or downloaded
GET /stems/{id} — to retrieve previously generated files
/reset	Clear session history
"""