import os
import tempfile
import numpy as np
from pydub import AudioSegment
import soundfile as sf

def numpy_array_to_audiosegment(array: np.ndarray, sr: int) -> AudioSegment:
    """
    Convert a numpy array to a pydub AudioSegment.

    Args:
        array: NumPy array with shape (channels, samples) - typically (2, N) for stereo
        sr: Sample rate in Hz (e.g., 44100)

    Returns:
        AudioSegment object that can be used with pydub operations
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        try:
            sf.write(temp_wav.name, array.T, sr)

            audio = AudioSegment.from_file(temp_wav.name, format="wav")

            return audio
        finally:
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)

