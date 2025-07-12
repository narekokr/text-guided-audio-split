import torchaudio
from demucs.demucs.pretrained import get_model
from demucs.demucs.apply import (apply_model)
import torchaudio.transforms as T
from pathlib import Path

def separate_audio(filepath: str, selected_stems: list[str]):
    model = get_model(name="mdx_extra_q")

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


    if not selected_stems:
        raise ValueError("No valid stems found in prompt. Please specify vocals, drums, bass, or other.")

    separated = apply_model(model, wav, device="cpu")  # Shape: [1, 4, 2, T]
    print(f"Model output shape: {separated.shape}")

    # Remove batch dim: shape becomes [4, 2, T]
    separated = separated[0]

    all_stems = ["drums", "bass", "other", "vocals"]

    filtered_stems = {
        stem_name: separated[i]
        for i, stem_name in enumerate(all_stems)
        if stem_name in selected_stems
    }

    return filtered_stems