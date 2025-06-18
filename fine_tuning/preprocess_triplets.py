import os
import yaml
import torch
import torchaudio
from pathlib import Path
from laion_clap import CLAP_Module

# --------- Paths & Config ---------
DATASET_PATH = Path("/mnt/c/Users/akyol/Desktop/data")
SAVE_PATH = Path("/mnt/c/Users/akyol/Desktop/triplets")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# --------- Model Setup ---------
clap_model = CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
clap_model.load_ckpt(
    '/home/akyol/clap-conditioned-source-separation-main/models/music_speech_audioset_epoch_15_esc_89.98.pt'
)
clap_model.eval()

# --------- Triplet Generator ---------
def create_triplet(track_path, stem_name_to_file):
    """
    For each stem in stem_name_to_file, embed a prompt including that stem_name,
    and save a triplet of (mix, stem, prompt, embedding).
    """
    mix_path = track_path / "mixture.wav"
    mix_waveform, sr = torchaudio.load(mix_path)

    if not stem_name_to_file:
        return

    for stem_name, stem_file in stem_name_to_file.items():
        # build output filename
        safe_name = stem_name.replace(' ', '_').replace('(', '').replace(')', '')
        output_file = SAVE_PATH / f"{track_path.name}_{safe_name}.pt"
        if output_file.exists():
            continue

        # Structured prompt: include only this stem
        structured_prompt = {"include": [stem_name]}
        prompt_str = str(structured_prompt)

        # CLAP embedding
        text_embed = clap_model.get_text_embedding([prompt_str], use_tensor=True)

        # Load stem waveform
        waveform, _ = torchaudio.load(stem_file)

        # Save triplet
        torch.save({
            "mix": mix_waveform,
            "stem": waveform,
            "prompt": prompt_str,
            "embedding": text_embed
        }, output_file)

        print(f"Saved: {output_file}")

# --------- Run for MUSDB18 ---------
def run_for_musdb18():
    musdb_train = DATASET_PATH / "musdb18/train"
    for track_dir in musdb_train.iterdir():
        if not track_dir.is_dir():
            continue

        stem_names = ["vocals", "drums", "bass", "other"]
        stem_files = {
            stem: track_dir / f"{stem}.wav"
            for stem in stem_names
            if (track_dir / f"{stem}.wav").exists()
        }
        if not stem_files:
            continue

        create_triplet(track_dir, stem_files)

# --------- Run for Slakh ---------
def run_for_slakh():
    slakh_train = DATASET_PATH / "slakh2100/train"
    for track_dir in slakh_train.iterdir():
        metadata_path = track_dir / "metadata.yaml"
        mix_flac = track_dir / "mix.flac"

        if not metadata_path.exists():
            continue

        # ensure mixture.wav exists
        mix_wav = track_dir / "mixture.wav"
        if not mix_wav.exists() and mix_flac.exists():
            waveform, sr = torchaudio.load(mix_flac)
            torchaudio.save(mix_wav, waveform, sr)
            try:
                mix_flac.unlink()
            except Exception:
                pass

        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        stems = metadata.get("stems", {})
        stem_dir = track_dir / "stems"

        stem_file_map = {}
        for stem_id, props in stems.items():
            if not props.get("audio_rendered", False):
                continue

            inst_class = props.get("inst_class", "").strip()
            midi_program = props.get("midi_program_name", "").strip()
            if not midi_program:
                continue

            # merge midi_program and inst_class
            merged_name = f"{midi_program} ({inst_class})" if inst_class else midi_program

            wav_flac = stem_dir / f"{stem_id}.flac"
            wav_path = wav_flac.with_suffix(".wav")
            if not wav_path.exists() and wav_flac.exists():
                waveform, sr = torchaudio.load(wav_flac)
                torchaudio.save(wav_path, waveform, sr)
            try:
                wav_flac.unlink()
            except Exception:
                pass

            if wav_path.exists():
                stem_file_map[merged_name] = wav_path

        if stem_file_map:
            create_triplet(track_dir, stem_file_map)

# --------- Main ---------
if __name__ == "__main__":
    run_for_musdb18()
    run_for_slakh()
