import os
import yaml
import torch
import torchaudio
from pathlib import Path
from laion_clap import CLAP_Module
import numpy as np

# --------- Paths & Config ---------
DATASET_PATH = Path("/mnt/c/Users/akyol/Desktop/data")
SAVE_PATH = Path("/mnt/c/Users/akyol/Desktop/triplets")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Segmenting params
SEGMENT_SEC = 5          # seconds per segment
STRIDE_SEC = 2.5         # seconds stride (overlap if < SEGMENT_SEC)
TARGET_SR = 44100        # sample rate
SILENCE_THRESHOLD = 1e-4 # RMS below this is considered silent

# --------- Model Setup ---------
clap_model = CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
clap_model.load_ckpt("/mnt/c/Users/akyol/Desktop/text-guided-audio-split-main/music_speech_audioset_epoch_15_esc_89.98.pt")
clap_model.eval()

def is_not_silent(waveform, threshold=SILENCE_THRESHOLD):
    """Returns True if waveform has RMS energy above threshold."""
    rms = waveform.pow(2).mean().sqrt().item()
    return rms > threshold

def create_triplet(track_path, stem_name_to_file, merged_names,
                   segment_sec=SEGMENT_SEC, stride_sec=STRIDE_SEC, target_sr=TARGET_SR):
    mix_path = track_path / "mixture.wav"
    mix_waveform, sr = torchaudio.load(mix_path)
    if sr != target_sr:
        mix_waveform = torchaudio.functional.resample(mix_waveform, sr, target_sr)
        sr = target_sr

    if not stem_name_to_file or not merged_names:
        return

    # Load all stems and stack: [S, C, T]
    stem_waveforms = []
    for stem_file in stem_name_to_file.values():
        s, s_sr = torchaudio.load(stem_file)
        if s_sr != target_sr:
            s = torchaudio.functional.resample(s, s_sr, target_sr)
        stem_waveforms.append(s)
    stem_waveforms = torch.stack(stem_waveforms)  # [S, C, T]
    num_stems, channels, total_len = stem_waveforms.shape

    segment_samples = int(segment_sec * sr)
    stride_samples = int(stride_sec * sr)
    num_segments = (total_len - segment_samples) // stride_samples + 1

    for seg_idx in range(num_segments):
        start = seg_idx * stride_samples
        end = start + segment_samples
        if end > total_len:
            break
        mix_seg = mix_waveform[:, start:end]           # [C, segment_samples]
        stems_seg = stem_waveforms[:, :, start:end]    # [S, C, segment_samples]

        # --- Ensure stereo ---
        if mix_seg.shape[0] == 1:
            mix_seg = mix_seg.repeat(2, 1)
        if stems_seg.shape[1] == 1:
            stems_seg = stems_seg.repeat(1, 2, 1)

        for stem_idx, (midi_program, merged_name) in enumerate(zip(stem_name_to_file.keys(), merged_names)):
            stem_seg = stems_seg[stem_idx]  # [C, segment_samples]
            if not is_not_silent(stem_seg):
                continue  # Skip if stem segment is silent

            text_embed = clap_model.get_text_embedding([merged_name], use_tensor=True)

            output_file = SAVE_PATH / f"{track_path.name}_{midi_program}_seg{seg_idx}.pt"
            np.savez_compressed(
                output_file.with_suffix('.npz'),
                mix=mix_seg.cpu().numpy(),
                stem=stem_seg.cpu().numpy(),
                text=np.array(merged_name),
                embedding=text_embed.detach().cpu().numpy()
            )

            print(f"Saved: {output_file}")

def run_for_slakh():
    slakh_train = DATASET_PATH / "slakh2100/train"
    for track_dir in slakh_train.iterdir():
        mix_flac = track_dir / "mix.flac"
        metadata_path = track_dir / "metadata.yaml"

        if not metadata_path.exists():
            continue

        mix_out = track_dir / "mixture.wav"
        if not mix_out.exists():
            waveform, sr = torchaudio.load(mix_flac)
            torchaudio.save(mix_out, waveform, sr)

        if mix_flac.exists():
            try:
                mix_flac.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {mix_flac}: {e}")

        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        stems = metadata.get("stems", {})
        stem_dir = track_dir / "stems"

        stem_file_map = {}
        name_list = []

        for stem_id, props in stems.items():
            if not props.get("audio_rendered", False):
                continue

            inst_class = props.get("inst_class", "")
            midi_program = props.get("midi_program_name", "")
            plugin_name = props.get("plugin_name", "")
            merged_name = f"{midi_program} ({inst_class})" if inst_class else midi_program
            if plugin_name is not None:
                merged_name += f" ({plugin_name})"

            if not midi_program:
                continue

            output_file = SAVE_PATH / f"{track_dir.name}_{midi_program}.pt"
            if output_file.exists():
                continue

            stem_path = stem_dir / f"{stem_id}.flac"
            wav_path = stem_path.with_suffix(".wav")

            if not wav_path.exists():
                waveform, sr = torchaudio.load(stem_path)
                torchaudio.save(wav_path, waveform, sr)

            if stem_path.exists():
                try:
                    stem_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {stem_path}: {e}")

            stem_file_map[midi_program] = wav_path
            name_list.append(merged_name)

        if stem_file_map:
            create_triplet(track_dir, stem_file_map, name_list)

if __name__ == "__main__":
    run_for_slakh()