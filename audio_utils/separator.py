import torch
import torchaudio
from pathlib import Path
import torchaudio.transforms as T

from demucs.demucs.pretrained import get_model
from demucs.demucs.apply import apply_model

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

def separate_audio(
    filepath: str,
    selected_stems: list[str],
    model_name: str = "hdemucs_mmi"
):
    """
    Separates audio into selected stems using CLAP conditioning per stem name.

    Args:
        filepath: path to input audio
        selected_stems: list of stems (["vocals", "drums"], etc.)
        model_name: Demucs model to use (must be CLAP-conditioned)
    Returns:
        Dict of {stem_name: audio_tensor}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = get_model(name=model_name)
    sources = ["drums", "bass", "other", "vocals"]
    model = HDemucs(sources=sources)
    model.to(device)

    try:
        torchaudio.set_audio_backend("sox_io")
    except RuntimeError:
        torchaudio.set_audio_backend("soundfile")

    assert Path(filepath).exists(), f"File not found: {filepath}"
    wav, sr = torchaudio.load(filepath)

    # [1, 2, T] (batched stereo)
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)
    if wav.shape[1] == 1:
        wav = wav.repeat(1, 2, 1)
    elif wav.shape[1] != 2:
        raise ValueError(f"Model requires stereo (2 channels), got shape: {wav.shape}")

    # Resample to 44.1kHz if needed
    if sr != 44100:
        resampler = T.Resample(orig_freq=sr, new_freq=44100)
        wav = resampler(wav)

    wav = wav.to(device)

    results = {}
    for stem in selected_stems:
        # 1. Get CLAP embedding of the stem name
        cond = get_clap_embedding(stem, device=device)  # [1, 512]
        
        # 2. Run model with conditioning
        with torch.no_grad():
            out = apply_model(model, wav, device=device, conditioning=cond)[0]  # [num_stems, 2, T]
        
        # 3. Find output index for this stem
        # Model's internal source order:
        model_stems = list(model.sources)
        if stem not in model_stems:
            continue  # ignore stems not in the model (just in case)
        i = model_stems.index(stem)
        results[stem] = out[i].cpu()

    return results