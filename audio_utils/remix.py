import soundfile as sf
import numpy as np
import uuid
from audio_utils.separator import separate_audio
from llm_backend.session_manager import get_file_from_db
import librosa
from pydub import AudioSegment
import os
import tempfile
from audio_utils.helpers import numpy_array_to_audiosegment



def handle_remix(intent: dict, session_id: str) -> dict:
    """
    Per-stem remix processing with support for both per-stem and global effects.
    """
    audio_path = get_file_from_db(session_id)
    if not audio_path:
        return {"reply": "No audio file found for remixing."}

    all_stems = ["vocals", "drums", "bass", "other"]
    outputs = separate_audio(audio_path, all_stems)

    stem_arrays = {}
    sr = 44100

    # Convert tensors to numpy arrays and ensure stereo
    for name in all_stems:
        tensor = outputs[name]
        if tensor.ndim == 3:
            tensor = tensor[0]   # shape [1, 2, N] → [2, N]
        array = tensor.numpy()
        if array.shape[0] == 1:  # mono to stereo
            array = np.repeat(array, 2, axis=0)
        stem_arrays[name] = array

    instructions = intent.get("instructions", {})
    volumes = instructions.get("volumes") or {}
    eq_instr = instructions.get("eq", {})
    filter_instr = instructions.get("filter", {})
    pitch_instr = instructions.get("pitch_shift", {})
    comp_instr = instructions.get("compression", {})
    reverb_instr = instructions.get("reverb", {})
    global_reverb = instructions.get("global_reverb", 0.0)

    min_len = min(arr.shape[1] for arr in stem_arrays.values())
    processed_segments = []

    # Process each stem individually with its effects
    for name, array in stem_arrays.items():
        array = array[:, :min_len]
        volume = volumes.get(name, 1.0)
        print(f"DEBUG - Processing {name}: volume={volume}")

        scaled = np.clip(array * volume, -1.0, 1.0)
        audio = numpy_array_to_audiosegment(scaled, sr)

        if name in reverb_instr and reverb_instr[name] > 0:
            audio = apply_reverb_pydub(audio, reverberance=reverb_instr[name])

        if name in pitch_instr:
            audio = change_pitch_pydub(audio, n_steps=pitch_instr[name])

        if name in eq_instr:
            eq = eq_instr[name]
            if all(k in eq for k in ("frequency", "width", "gain_db")):
                audio = apply_eq_pydub(audio, frequency=eq["frequency"], width=eq["width"], gain_db=eq["gain_db"])

        if name in filter_instr:
            f = filter_instr[name]
            if f["type"] == "lowpass" and "cutoff" in f:
                audio = apply_filter(audio, "lowpass", cutoff=f["cutoff"])
            elif f["type"] == "highpass" and "cutoff" in f:
                audio = apply_filter(audio, "highpass", cutoff=f["cutoff"])
            elif f["type"] == "bandpass" and "low_cutoff" in f and "high_cutoff" in f:
                audio = apply_filter(audio, "highpass", cutoff=f["low_cutoff"])
                audio = apply_filter(audio, "lowpass", cutoff=f["high_cutoff"])

        if name in comp_instr:
            level = comp_instr[name]
            if level == "low":
                audio = apply_compression_pydub(audio, threshold=-30, ratio=2)
            elif level == "medium":
                audio = apply_compression_pydub(audio, threshold=-20, ratio=4)
            elif level == "high":
                audio = apply_compression_pydub(audio, threshold=-10, ratio=8)

        processed_segments.append(audio)
        print(f"DEBUG - Added {name} to processed segments (duration: {len(audio)}ms)")

    if not processed_segments:
        return {"reply": "No stems were processed for remixing."}

    print(f"DEBUG - Mixing {len(processed_segments)} processed segments")
    final_mix = processed_segments[0]
    for i, seg in enumerate(processed_segments[1:], 1):
        print(f"DEBUG - Overlaying segment {i+1}")
        final_mix = final_mix.overlay(seg)

    if global_reverb > 0:
        print(f"DEBUG - Applying global reverb: {global_reverb}")
        final_mix = apply_reverb_pydub(final_mix, reverberance=global_reverb)

    output_name = generate_remix_name(intent)
    output_path = f"separated/{output_name}"
    print(f"DEBUG - Exporting final mix to: {output_path}")
    final_mix.export(output_path, format="wav")
    print(f"DEBUG - Export completed. Final mix duration: {len(final_mix)}ms")

    return {
        "reply": f"Remix is created based on instructions.",
        "remix": {"file_url": f"/downloads/{output_name}"}
    }

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

                reverb_level = max(0.0, min(reverberance, 1.0))  # Clamp to 0-1
                delay_ms = int(reverb_level * 100 + 50)  # 50-150ms delay
                decay = reverb_level * 0.6  # 0-0.6 decay

                print(f"DEBUG - Applying reverb: level={reverb_level}, delay={delay_ms}ms, decay={decay}")

                from pydub.utils import which
                ffmpeg_path = which("ffmpeg")

                import subprocess
                cmd = [
                    ffmpeg_path, "-i", temp_input.name,
                    "-af", f"aecho=0.8:0.9:{delay_ms}:{decay}",
                    "-y", temp_output.name
                ]

                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"DEBUG - Reverb ffmpeg completed successfully")

                return AudioSegment.from_wav(temp_output.name)

            except subprocess.CalledProcessError as e:
                print(f"ERROR - Reverb ffmpeg failed: {e.stderr}")
                return audio  # Return original audio if reverb fails
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
                attack_sec = attack / 1000.0  # ms to seconds
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
    Join descriptors with underscores
    type_str = "_".join(remix_type) if remix_type else "basic"
    output_name = f" remix_{type_str}_{uuid.uuid4().hex[:6]}.wav"
    """

    remix_type = []

    volumes = intent["instructions"].get("volumes")
    if volumes:
        remix_type.append("vol")

    for effect in ["reverb", "pitch_shift", "compression"]:
        effect_dict = intent["instructions"].get(effect)
        if effect_dict and any(effect_dict.values()):
            remix_type.append(effect.replace("_", ""))

    # Check for global reverb
    if intent["instructions"].get("global_reverb", 0.0) > 0:
        remix_type.append("globalreverb")

    type_str = "_".join(remix_type) if remix_type else "basic"
    output_name = f"remix_{type_str}_{uuid.uuid4().hex[:6]}.wav"
    return output_name

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
