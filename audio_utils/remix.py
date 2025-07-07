import soundfile as sf
import numpy as np
import uuid
from audio_utils.separator import separate_audio
from llm_backend.session_manager import get_file_from_db
import torchaudio
import librosa
from pydub import AudioSegment
import os
import tempfile

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
    print(f"DEBUG - Full intent: {intent}")
    print(f"DEBUG - Instructions: {instructions}")
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
    print(f"[DEBUG] Naming remix with intent: {intent}")
    output_name = generate_remix_name(intent)
    output_path = f"separated/{output_name}"

    sf.write(output_path, mix.T, sr)  # Note transpose to (samples, channels)
    audio = AudioSegment.from_wav(output_path)

    reverb_instr = intent["instructions"].get("reverb", {})
    print(f"DEBUG - Reverb instructions: {reverb_instr}")

    if reverb_instr:
        for stem, amount in reverb_instr.items():
            if stem in all_stems and amount > 0:
                #output_path = apply_reverb(output_path, output_path, reverberance=amount * 100)  # scale if needed
                audio = apply_reverb_pydub(audio, reverberance=amount)

    pitch_instr = intent["instructions"].get("pitch_shift", {})
    print(f"DEBUG - Pitch instructions: {pitch_instr}")

    if pitch_instr:
        for stem, semitones in pitch_instr.items():
            if stem in all_stems:
                #output_path = change_pitch_pydub(audio, n_steps=semitones) #(output_path, output_path, n_steps=semitones)
                audio = change_pitch_pydub(audio, n_steps=semitones)

    #compression
    comp_instr = intent["instructions"].get("compression", {})
    print(f"DEBUG - Compression instructions: {comp_instr}")

    if comp_instr:
        for stem, level in comp_instr.items():
            if stem in all_stems:
                if level == "low":
                    audio = apply_compression_pydub(audio, threshold=-30, ratio=2)
                elif level == "medium":
                    audio = apply_compression_pydub(audio, threshold=-20, ratio=4)
                elif level == "high":
                    audio = apply_compression_pydub(audio, threshold=-10, ratio=8)
    audio.export(output_path, format="wav")
    return {
        "reply": f"Remix created based on instructions.",
        "remix": {"file_url": f"/downloads/{output_name}"}
    }

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
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
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


def apply_gain_scaling(stem_arrays, volumes, min_len):
    adjusted_stems = []
    for name, array in stem_arrays.items():
        scaled = np.clip(array[:, :min_len] * volumes.get(name, 1.0), -1.0, 1.0)
        adjusted_stems.append(scaled)
    return adjusted_stems


def apply_reverb_pydub(audio: AudioSegment, reverberance: float = 50.0):
    """
    Apply reverb using ffmpeg through pydub
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            try:
                # Export to temp file
                audio.export(temp_input.name, format="wav")

                # Apply reverb using ffmpeg
                # Scale reverberance to appropriate values for ffmpeg
                reverb_level = min(reverberance / 100.0, 1.0)  # Convert to 0-1 range

                # Use ffmpeg's aecho filter for reverb-like effect
                from pydub.utils import which
                ffmpeg_path = which("ffmpeg")

                import subprocess
                cmd = [
                    ffmpeg_path, "-i", temp_input.name,
                    "-af", f"aecho=0.8:0.88:{reverb_level * 60}:{reverb_level * 0.4}",
                    "-y", temp_output.name
                ]

                subprocess.run(cmd, check=True, capture_output=True)

                # Load processed audio
                return AudioSegment.from_wav(temp_output.name)

            finally:
                # Clean up temp files
                for temp_file in [temp_input.name, temp_output.name]:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)


def change_pitch_pydub(audio: AudioSegment, n_steps: float):
    """
    Change pitch using librosa (more reliable than ffmpeg for pitch shifting)
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        try:
            audio.export(temp_file.name, format="wav")

            # Use librosa for pitch shifting (more reliable)
            y, sr = librosa.load(temp_file.name, sr=None)
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

            # Convert back to AudioSegment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                sf.write(temp_output.name, y_shifted, sr)
                return AudioSegment.from_wav(temp_output.name)

        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            if 'temp_output' in locals() and os.path.exists(temp_output.name):
                os.unlink(temp_output.name)


def apply_compression_pydub(audio: AudioSegment, threshold: float = -20, ratio: float = 3,
                            attack: float = 20, release: float = 250):
    """
    Apply compression using ffmpeg through pydub
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            try:
                # Export to temp file
                audio.export(temp_input.name, format="wav")

                # Apply compression using ffmpeg
                from pydub.utils import which
                ffmpeg_path = which("ffmpeg")

                import subprocess
                # Convert parameters for ffmpeg compand filter
                attack_sec = attack / 1000.0  # Convert ms to seconds
                release_sec = release / 1000.0

                cmd = [
                    ffmpeg_path, "-i", temp_input.name,
                    "-af",
                    f"compand=attacks={attack_sec}:decays={release_sec}:points={threshold}/{threshold}|0/{threshold - threshold / ratio}",
                    "-y", temp_output.name
                ]

                subprocess.run(cmd, check=True, capture_output=True)

                # Load processed audio
                return AudioSegment.from_wav(temp_output.name)

            finally:
                # Clean up temp files
                for temp_file in [temp_input.name, temp_output.name]:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)

def generate_remix_name(intent: dict) -> str:
    """
    remix_type = []
    if volumes:
        remix_type.append("vol")

    if intent["instructions"].get("reverb") and any(intent["instructions"]["reverb"].values()):
        remix_type.append("reverb")

    if intent["instructions"].get("pitch_shift") and any(intent["instructions"]["pitch_shift"].values()):
        remix_type.append("pitch")

    if intent["instructions"].get("compression") and any(intent["instructions"]["compression"].values()):
        remix_type.append("comp")

    # Join descriptors with underscores
    type_str = "_".join(remix_type) if remix_type else "basic"
    output_name = f"remix_{type_str}_{uuid.uuid4().hex[:6]}.wav"
    """
    remix_type = []

    volumes = intent["instructions"].get("volumes")
    if volumes:
        remix_type.append("vol")

    for effect in ["reverb", "pitch_shift", "compression"]:
        effect_dict = intent["instructions"].get(effect)
        if effect_dict and any(effect_dict.values()):
            remix_type.append(effect.replace("_", ""))  # e.g. pitch_shift → pitch

    type_str = "_".join(remix_type) if remix_type else "basic"
    output_name = f"remix_{type_str}_{uuid.uuid4().hex[:6]}.wav"
    return output_name


"""
#Note:
sample prompts for testing: add heavy reverb to vocals, pitch vocal up by 2 semitones, compress vocals to make them tighter
Reverb and pitch shifting do not require equal lengths individually, but when we mix stems back together (e.g., vocals + drums) they must match in sample length. 
"""