import torchaudio
from demucs.demucs.pretrained import get_model
from demucs.demucs.apply import (apply_model)

from llm_backend.interpreter import interpret_prompt

def separate_audio(filepath: str, prompt: str):
    model = get_model(name="mdx_extra_q")
    wav, sr = torchaudio.load(filepath)
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)

    #separate into all stems
    stems = apply_model(model, wav, device="cpu")

    #interpret prompt
    selected_stems = interpret_prompt(prompt)
    #return dictionary with keys applicable to demucs
    filtered_stems = {name: audio for name, audio in stems.items() if name in selected_stems}
    return filtered_stems