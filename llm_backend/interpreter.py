import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model(name="mdx_extra_q")
waveform, sr = torchaudio.load("your_audio.wav")

#pretrained model mdx_extra_q is always trained to output those 4 stems

sources = apply_model(model, waveform, device="cpu")
# sources is a dict like: { 'vocals': tensor, 'drums': tensor, ... }

def interpret():
    return 0