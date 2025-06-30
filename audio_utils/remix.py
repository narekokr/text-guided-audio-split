import soundfile as sf
import numpy as np
import uuid
from audio_utils.separator import separate_audio
from llm_backend.session_manager import get_file_from_db
import torchaudio
import torchaudio.sox_effects as sox_effects
import librosa
import soundfile

def handle_remix(intent: dict, session_id: str) -> dict:
    audio_path = get_file_from_db(session_id) #retrieves audio file associated with this session
    if not audio_path:
        return {"reply": " No audio file found for remixing."}

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

    instructions = intent.get("instructions", {})
    volumes = instructions.get("volumes", {})

    min_len = min(arr.shape[1] for arr in stem_arrays.values())
    # Only apply gain scaling if volumes were specified

    if volumes:
        adjusted_stems = apply_gain_scaling(stem_arrays, volumes, min_len)
    else:
        # Default: use original stems truncated to min_len
        adjusted_stems = [arr[:, :min_len] for arr in stem_arrays.values()]

    # Truncate and mix
    #stem_arrays[name][:, :min_len] - retrieves the stem waveform
    #volumes.get(name, 1.0) retrieves the volume multiplier, defaults to 1.0 if not specified
    #multiplying these ensure that waveform is scaled by this factor, which is gain scaling, i.e. volume adjustment

    mix = sum(adjusted_stems)

    """
         = sum(
            np.clip(stem_arrays[name][:, :min_len] * volumes.get(name, 1.0), -1.0, 1.0)
            for name in all_stems
        )
    """
    mix = np.clip(mix, -1.0, 1.0) #to ensure that combined mix does not exceed the allowed amplitude limits
    output_name = f"remix_{uuid.uuid4().hex[:6]}.wav"
    output_path = f"separated/{output_name}"
    sf.write(output_path, mix.T, sr)  # Note transpose to (samples, channels)

    #apply reverb if instruction is directing that
    reverb_instr = intent["instructions"].get("reverb", {})
    for stem, amount in reverb_instr.items():
        if stem in all_stems:
            output_path = apply_reverb(output_path, output_path, reverberance=amount * 100)  # scale if needed

    pitch_instr = intent["instructions"].get("pitch_shift", {})
    for stem, semitones in pitch_instr.items():
        if stem in all_stems:
            output_path = change_pitch(output_path, output_path, n_steps=semitones)

    #compression
    comp_instr = intent["instructions"].get("compression", {})
    for stem, level in comp_instr.items():
        if stem in all_stems:
            # Map 'low', 'medium', 'high' to parameters
            if level == "low":
                output_path = apply_compression(output_path, output_path, threshold=-30, ratio=2)
            elif level == "medium":
                output_path = apply_compression(output_path, output_path, threshold=-20, ratio=4)
            elif level == "high":
                output_path = apply_compression(output_path, output_path, threshold=-10, ratio=8)

    return {
        "reply": f"Remix created based on instructions.",
        "remix": {"file_url": f"/downloads/{output_name}"}
    }


def apply_gain_scaling(stem_arrays, volumes, min_len):
    adjusted_stems = []
    for name, array in stem_arrays.items():
        scaled = np.clip(array[:, :min_len] * volumes.get(name, 1.0), -1.0, 1.0)
        adjusted_stems.append(scaled)
    return adjusted_stems

def apply_reverb(input_path: str, output_path: str, reverberance: float = 50.0):
    effects = [
        ["reverb", str(reverberance)]
    ]
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(input_path, effects)

    torchaudio.save(output_path, waveform, sample_rate)
    return output_path


def change_pitch(input_path: str, output_path: str, n_steps: float):
    """
    n_steps: positive to increase pitch, negative to decrease (in semitones)
    """
    y, sr = librosa.load(input_path, sr=None)
    y_shifted = librosa.effects.pitch_shift(y, sr, n_steps)
    sf.write(output_path, y_shifted, sr)
    return output_path

def apply_compression(input_path: str, output_path: str, attack=20, release=250, threshold=-20, ratio=3):
    """
    attack: attack time in ms
    release: release time in ms
    threshold: threshold in dB
    ratio: compression ratio
    """
    effects = [
        ["compand", f"{attack},{release}", f"{threshold},{threshold},{ratio}", "-90", "-90", "0", "-90", "0.2"]
    ]
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(input_path, effects)
    torchaudio.save(output_path, waveform, sample_rate)
    return output_path

"""
#Note:
sample prompts for testing: add heavy reverb to vocals, pitch vocal up by 2 semitones, compress vocals to make them tighter
Reverb and pitch shifting do not require equal lengths individually, but when we mix stems back together (e.g., vocals + drums) they must match in sample length. 
"""