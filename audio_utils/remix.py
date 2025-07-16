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


session_last_instructions = {}
session_active_task = {}

def handle_remix(intent: dict, session_id: str) -> dict:
    audio_path = get_file_from_db(session_id)
    if not audio_path:
        return {"reply": " No audio file found for remixing."}

    instructions = intent.get("instructions", {})
    # All stems the user wants to change
    requested_stems = set()
    for param in ["volumes", "reverb", "pitch_shift", "compression", "eq", "filter"]:
        requested_stems.update(instructions.get(param, {}).keys())
    requested_stems = list(requested_stems)
    if not requested_stems:
        return {"reply": "No stems specified for remixing."}

    # Separate ONLY the requested stems
    outputs = separate_audio(audio_path, requested_stems)  # returns {stem_name: tensor}

    sr = 44100
    orig_mix, orig_sr = sf.read(audio_path, dtype="float32")
    if orig_mix.ndim == 1:
        orig_mix = np.stack([orig_mix, orig_mix])
    elif orig_mix.shape[1] == 1:
        orig_mix = np.repeat(orig_mix, 2, axis=1)
    orig_mix = orig_mix.T  # (channels, samples)
    min_len = orig_mix.shape[1]

    # Prepare separated stem arrays, ensure stereo
    separated = {}
    for name in requested_stems:
        tensor = outputs[name]
        if tensor.ndim == 3:
            tensor = tensor[0]
        arr = tensor.numpy()
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 2, axis=0)
        separated[name] = arr[:, :min_len]

    # 1. Rebuild the sum of all separated stems
    sum_of_separated = np.zeros_like(orig_mix)
    for arr in separated.values():
        sum_of_separated += arr[:, :min_len]

    # 2. Subtract the separated stems from the original mix to get the "remainder" mix
    remainder = orig_mix[:, :min_len] - sum_of_separated

    # 3. Process the requested stems as per user instructions
    processed = {}
    for stem_name, arr in separated.items():
        # Convert to wav temp for pydub-based effects
        temp_fn = f"temp_{uuid.uuid4().hex[:6]}_{stem_name}.wav"
        sf.write(temp_fn, arr.T, sr)
        audio = AudioSegment.from_wav(temp_fn)
        os.remove(temp_fn)

        # Gain (volume)
        gain = instructions.get("volumes", {}).get(stem_name, 1.0)
        arr = np.clip(arr * gain, -1.0, 1.0)

        # Reverb
        if stem_name in instructions.get("reverb", {}):
            audio = apply_reverb_pydub(audio, reverberance=instructions["reverb"][stem_name])
        # EQ
        if stem_name in instructions.get("eq", {}):
            eq = instructions["eq"][stem_name]
            freq = eq.get("frequency")
            width = eq.get("width")
            gain_db = eq.get("gain_db")
            if freq and width and gain_db is not None:
                audio = apply_eq_pydub(audio, frequency=freq, width=width, gain_db=gain_db)
        # Filter
        if stem_name in instructions.get("filter", {}):
            f = instructions["filter"][stem_name]
            ftype = f.get("type")
            if ftype in ("lowpass", "highpass") and f.get("cutoff"):
                audio = apply_filter(audio, filter_type=ftype, cutoff=f["cutoff"])
            elif ftype == "bandpass":
                if f.get("low_cutoff") and f.get("high_cutoff"):
                    audio = apply_filter(audio, "highpass", cutoff=f["low_cutoff"])
                    audio = apply_filter(audio, "lowpass", cutoff=f["high_cutoff"])
        # Pitch shift
        if stem_name in instructions.get("pitch_shift", {}):
            audio = change_pitch_pydub(audio, n_steps=instructions["pitch_shift"][stem_name])
        # Compression
        if stem_name in instructions.get("compression", {}):
            level = instructions["compression"][stem_name]
            if level == "low":
                audio = apply_compression_pydub(audio, threshold=-30, ratio=2)
            elif level == "medium":
                audio = apply_compression_pydub(audio, threshold=-20, ratio=4)
            elif level == "high":
                audio = apply_compression_pydub(audio, threshold=-10, ratio=8)
        # Back to numpy
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
            audio.export(temp_out.name, format="wav")
            y, _ = sf.read(temp_out.name, dtype="float32")
            if y.ndim == 1:
                y = np.stack([y, y])
            elif y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            y = y.T  # (channels, samples)
            processed[stem_name] = y[:, :min_len]
            os.remove(temp_out.name)

    # 4. Rebuild the final mix: processed stems + untouched remainder
    final_mix = remainder.copy()
    for arr in processed.values():
        final_mix += arr[:, :min_len]
    final_mix = np.clip(final_mix, -1.0, 1.0)

    # 5. Save and return
    output_name = generate_remix_name(intent)
    output_path = f"separated/{output_name}"
    sf.write(output_path, final_mix.T, sr)
    session_last_instructions[session_id] = instructions

    return {
        "reply": f"Remix is created based on instructions.",
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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            try:
                audio.export(temp_input.name, format="wav")

                # Scale reverberance to appropriate values for ffmpeg
                reverb_level = min(reverberance / 100.0, 1.0)  # Convert to 0-1 range

                from pydub.utils import which
                ffmpeg_path = which("ffmpeg")

                import subprocess
                cmd = [
                    ffmpeg_path, "-i", temp_input.name,
                    "-af", f"aecho=0.8:0.88:{reverb_level * 60}:{reverb_level * 0.4}",
                    "-y", temp_output.name
                ]

                subprocess.run(cmd, check=True, capture_output=True)

                return AudioSegment.from_wav(temp_output.name)

            finally:
                for temp_file in [temp_input.name, temp_output.name]:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)


def change_pitch_pydub(audio: AudioSegment, n_steps: float):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        try:
            audio.export(temp_file.name, format="wav")

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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            try:
                audio.export(temp_input.name, format="wav")

                from pydub.utils import which
                ffmpeg_path = which("ffmpeg")

                import subprocess
                attack_sec = attack / 1000.0  # Convert ms to seconds
                release_sec = release / 1000.0

                cmd = [
                    ffmpeg_path, "-i", temp_input.name,
                    "-af",
                    f"compand=attacks={attack_sec}:decays={release_sec}:points={threshold}/{threshold}|0/{threshold - threshold / ratio}",
                    "-y", temp_output.name
                ]

                subprocess.run(cmd, check=True, capture_output=True)
                return AudioSegment.from_wav(temp_output.name)

            finally:
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
Boost 100 Hz for bass,

Cut 2 kHz to reduce harshness in vocals,

Boost 12 kHz to add "air" or brightness.
"""

def apply_eq_pydub(audio: AudioSegment, frequency: float, width: float, gain_db: float):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            try:
                audio.export(temp_input.name, format="wav")

                from pydub.utils import which
                ffmpeg_path = which("ffmpeg")
                import subprocess

                cmd = [
                    ffmpeg_path, "-i", temp_input.name,
                    "-af", f"equalizer=f={frequency}:t=q:w={width}:g={gain_db}",
                    "-y", temp_output.name
                ]

                subprocess.run(cmd, check=True, capture_output=True)
                return AudioSegment.from_wav(temp_output.name)

            finally:
                for temp_file in [temp_input.name, temp_output.name]:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)

def apply_filter(audio: AudioSegment, filter_type="lowpass", cutoff=5000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            try:
                audio.export(temp_input.name, format="wav")
                from pydub.utils import which
                ffmpeg_path = which("ffmpeg")

                import subprocess

                if filter_type == "lowpass":
                    af = f"lowpass=f={cutoff}"
                elif filter_type == "highpass":
                    af = f"highpass=f={cutoff}"
                else:
                    raise ValueError("Unsupported filter type.")

                cmd = [
                    ffmpeg_path, "-i", temp_input.name,
                    "-af", af,
                    "-y", temp_output.name
                ]

                subprocess.run(cmd, check=True, capture_output=True)
                return AudioSegment.from_wav(temp_output.name)

            finally:
                for temp_file in [temp_input.name, temp_output.name]:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
