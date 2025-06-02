import torch
import torchaudio
from demucs.demucs.pretrained import get_model
from demucs.demucs.apply import apply_model

def separate_audio(filepath: str):
    """
    Separates an input audio file into individual stems using a pretrained Demucs model.
    Parameters:
        filepath (str): Path to the input audio file (.wav format) to be processed.

    Returns:
        dict[str, torch.Tensor]: A dictionary mapping stem names (e.g., 'vocals', 'drums', 'bass', 'other')
        to their corresponding separated audio waveforms as PyTorch tensors.

    Notes:
        - The Demucs model used is 'mdx_extra_q', which is trained to separate 4 stems.
        - Input audio is in .wav format.
        - The function runs on CPU by default for compatibility.
    """
    model = get_model(name="mdx_extra_q")
    wav, sr = torchaudio.load(filepath)
    stems = apply_model(model, wav, device="cpu")
    return stems


def filter_stems(all_stems: dict[str, torch.Tensor], selected: list[str]) -> dict[str, torch.Tensor]:
    """
    Filters the separated stems to include only the selected ones.
    Can be used in main to filter if requested stems are already present.
    Parameters:
        all_stems (dict): Dictionary of all separated stems.
        selected (list[str]): List of stem names to retain (e.g., ['vocals', 'drums']).

    Returns:
        dict: Filtered dictionary containing only selected stems.
    """
    return {name: audio for name, audio in all_stems.items() if name in selected}


def mix_stems(stems: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Mixes selected audio stems into a single waveform.

    Parameters:
        stems (dict): Dictionary of selected stem tensors.

    Returns:
        torch.Tensor: Combined audio tensor if user requests several stems at once
    """
    if not stems:
        raise ValueError("No stems provided for mixing.")

    stem_values = list(stems.values())
    mixed = torch.zeros_like(stem_values[0]) #a zero-tensor with same shape and dtype as the first stem to ensure that accumulation is possible
    for stem in stem_values:
        mixed += stem
    return mixed

