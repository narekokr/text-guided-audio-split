import torchaudio
from demucs.demucs.pretrained import get_model
from demucs.demucs.apply import (apply_model)
import torchaudio.transforms as T
import torch
from llm_backend.interpreter import interpret_prompt

"""

def separate_audio(filepath: str, prompt: str):
    model = get_model(name="mdx_extra_q")
    wav, sr = torchaudio.load(filepath)
    #demucs requires  (1, 2, T) tensor
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)

    if wav.shape[1] == 1:
        # Mono audio — repeat channels to get stereo
        wav = wav.repeat(1, 2, 1)

    elif wav.shape[1] != 2:
        raise ValueError(f"Demucs requires stereo (2 channels), but got shape: {wav.shape}")

    resampler = T.Resample(orig_freq=sr, new_freq=44100) #sample rate needs to be 44.1 hz
    wav = resampler(wav)

    selected_stems = interpret_prompt(prompt)

    if not selected_stems:
        raise ValueError("No valid stems found in prompt. Please specify vocals, drums, bass, or other.")

    #separate into all stems
    stems = apply_model(model, wav, device="cpu")
    #TODO  map the indices to stem names, and then filter based on selected_stems
    all_stems = ["drums", "bass", "other", "vocals"]

    # stems shape: [1, 4, 2, N]
    stems = stems[0]  # remove batch dimension

    filtered_stems = {
        stem_name: stems[i]
        for i, stem_name in enumerate(all_stems)
        if stem_name in selected_stems
    }
    #this should return something like:  "drums": <Tensor [2, N]>, which will work with base63 encoding i
    return filtered_stems

"""

def separate_audio(filepath: str, prompt: str):
    model = get_model(name="mdx_extra_q")
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

    # Interpret the prompt (e.g., "vocals and drums") -> ['vocals', 'drums']
    selected_stems = interpret_prompt(prompt)

    if not selected_stems:
        raise ValueError("No valid stems found in prompt. Please specify vocals, drums, bass, or other.")

    # Apply Demucs separation
    separated = apply_model(model, wav, device="cpu")  # Shape: [1, 4, 2, T]
    print(f"Model output shape: {separated.shape}") #1, 4, 2, N if this is not hte case we might need to switch to model = get_model(name="htdemucs")  # known 4-stem model

    # Remove batch dim: shape becomes [4, 2, T]
    separated = separated[0]

    # Demucs standard stem order
    all_stems = ["drums", "bass", "other", "vocals"]

    # Map and filter the stems
    filtered_stems = {
        stem_name: separated[i]
        for i, stem_name in enumerate(all_stems)
        if stem_name in selected_stems
    }

    return filtered_stems
