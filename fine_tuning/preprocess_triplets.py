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
clap_model.load_ckpt('/home/akyol/clap-conditioned-source-separation-main/models/music_speech_audioset_epoch_15_esc_89.98.pt')
clap_model.eval()

# --------- Triplet Generator ---------
def create_triplet(track_path, stem_name_to_file, inst_classes):
    mix_path = track_path / "mixture.wav"
    mix_waveform, sr = torchaudio.load(mix_path)

    if not stem_name_to_file:
        return

    for stem_name, stem_file in stem_name_to_file.items():
        output_file = SAVE_PATH / f"{track_path.name}_{stem_name}.pt"
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
        if track_dir.is_dir():
            stem_names = ["vocals", "drums", "bass", "other"]
            stem_files = {
                stem: track_dir / f"{stem}.wav"
                for stem in stem_names
                if (track_dir / f"{stem}.wav").exists()
            }
            if not stem_files:
                continue
            create_triplet(track_dir, stem_files, list(stem_files.keys()))

# --------- Run for Slakh ---------
def run_for_slakh():
    slakh_train = DATASET_PATH / "slakh2100/train"
    for track_dir in slakh_train.iterdir():
        mix_flac = track_dir / "mix.flac"
        metadata_path = track_dir / "metadata.yaml"

        if not mix_flac.exists() or not metadata_path.exists():
            continue

        mix_out = track_dir / "mixture.wav"
        if not mix_out.exists():
            waveform, sr = torchaudio.load(mix_flac)
            torchaudio.save(mix_out, waveform, sr)

        # Always try to delete mix.flac if it exists
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
            merged_name = f"{midi_program} ({inst_class})" if inst_class else midi_program

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

            # Always try to delete the .flac even if .wav exists
            if stem_path.exists():
                try:
                    stem_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {stem_path}: {e}")

            stem_file_map[midi_program] = wav_path
            name_list.append(merged_name)

        if stem_file_map:
            create_triplet(track_dir, stem_file_map, name_list)

# --------- Main ---------
if __name__ == "__main__":
    run_for_musdb18()
    run_for_slakh()
