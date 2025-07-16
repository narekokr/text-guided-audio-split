import torchaudio
from demucs.demucs.pretrained import get_model
from demucs.demucs.apply import (apply_model)
import torchaudio.transforms as T
from pathlib import Path

from demucs.demucs.hdemucs import HDemucs
from laion_clap import CLAP_Module

# Global CLAP model cache
_clap_model = None
def get_clap_model(device="cpu"):
    global _clap_model
    if _clap_model is None and CLAP_Module is not None:
        _clap_model = CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        _clap_model.load_ckpt(
            'C:/Users/akyol/Desktop/text-guided-audio-split-main/music_speech_audioset_epoch_15_esc_89.98.pt'
        )   # Update this path as needed
        _clap_model.eval()
       # _clap_model.to(device)
    return _clap_model

def get_clap_embedding(text: str, device="cpu"):
    model = get_clap_model(device=device)
    if model is None:
        raise RuntimeError("CLAP_Module not available")
    with torch.no_grad():
        emb = model.get_text_embedding([text], use_tensor=True).to(device)
    return emb  # [1, 512]

def separate_audio(filepath: str, selected_stems: list[str]):    
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    #model = get_model(name="mdx_extra_q")
    sources = ["stem"]
    model = HDemucs(sources = sources)
    checkpoint = torch.load("outputs/xps/97d170e1/best.th", map_location="cpu")
    model.load_state_dict(checkpoint['state'])
    model.eval()
    model.to(device)
    
    try:
        torchaudio.set_audio_backend("sox_io")
    except RuntimeError:
        torchaudio.set_audio_backend("soundfile")

    print("Resolving file:", Path(filepath).resolve())
    assert Path(filepath).exists(), f"File not found: {filepath}"
    wav, sr = torchaudio.load(filepath)

    # Ensure the audio is batched: shape [1, channels, time]
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)

    # Convert mono to stereo if needed
    if wav.shape[1] == 1:
        wav = wav.repeat(1, 2, 1)
    elif wav.shape[1] != 2:
        raise ValueError(f"Demucs requires stereo (2 channels), but got shape: {wav.shape}")

    print(f"WAV shape: {wav.shape}, sample rate: {sr}")

    # Resample if not 44.1kHz
    if sr != 44100:
        resampler = T.Resample(orig_freq=sr, new_freq=44100)
        wav = resampler(wav)

    results = {}
    for stem in selected_stems:
        # 1. Get CLAP embedding of the stem name
        cond = get_clap_embedding(stem, device=device)  # [1, 512]

        # 2. Run model with conditioning
        with torch.no_grad():
            out = apply_model(model, wav, device=device, conditioning=cond)  # [2, T] or [1, 2, T]
            # Remove batch dim if needed
            if out.ndim == 3 and out.shape[0] == 1:
                out = out[0]
        results[stem] = out.cpu()

    return results