#for combining, selecting and filtering stems

import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

def separate_audio(filepath: str):
    model = get_model(name="mdx_extra_q")
    wav, sr = torchaudio.load(filepath)
    stems = apply_model(model, wav, device="cpu")
    return stems  # dict with keys: vocals, drums, etc