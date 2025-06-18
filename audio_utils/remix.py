import soundfile as sf
import numpy as np
import torchaudio
import os
import uuid
from audio_utils.separator import separate_audio
from llm_backend.chat_manager import session_manager

#Manipulation (gain scaling) - with numpy it is possible to do this without manipulating tensors manually
def handle_remix(intent: dict, session_id: str) -> dict:
    audio_path = session_manager.get_file(session_id) #retrieves audio file associated with this session
    if not audio_path:
        return {"reply": " No audio file found for remixing."}

    #Separate all 4 stems
    all_stems = ["vocals", "drums", "bass", "other"]
    outputs = separate_audio(audio_path, all_stems)

    stem_arrays = {} # dictionary to hold NumPy arrays for each stem, and sets the sample rate to 44.1kHz
    sr = 44100

    for name in all_stems:
        tensor = outputs[name]
        if tensor.ndim == 3:
            tensor = tensor[0]   # shape [1, 2, N] → [2, N] or [1, N] for channels, samples
        array = tensor.numpy()
        if array.shape[0] == 1:  # mono to stereo
            array = np.repeat(array, 2, axis=0) #this way we ensure we have shape [2, samples] and it becomes mixable even if some stems started as mono
        stem_arrays[name] = array #Stores the processed stereo waveform in the dictionary under its stem name.

    # Step 2: Remix with volume multipliers
    volumes = intent["volumes"]
    min_len = min(arr.shape[1] for arr in stem_arrays.values())

    # Truncate and mix
    mix = sum(
        np.clip(stem_arrays[name][:, :min_len] * volumes.get(name, 1.0), -1.0, 1.0)
        for name in all_stems
    )

    mix = np.clip(mix, -1.0, 1.0)
    output_name = f"remix_{uuid.uuid4().hex[:6]}.wav"
    output_path = f"separated/{output_name}"
    sf.write(output_path, mix.T, sr)  # Note transpose to (samples, channels)

    return {
        "reply": f"Remix created based on instructions.",
        "remix": {"file_url": f"/downloads/{output_name}"}
    }
