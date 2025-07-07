import os
import numpy as np
import soundfile as sf
import pytest
from llm_backend.session_manager import save_file_to_db  # hypothetical function
from audio_utils.remix import handle_remix
from audio_utils.remix import apply_gain_scaling, apply_reverb, change_pitch, apply_compression
from unittest.mock import patch, MagicMock
TEST_AUDIO = "tests/test.wav"

@patch("llm_backend.session_manager.save_file_to_db")
def test_handle_remix_flow(mock_save, tmp_path):
    # Setup dummy file in DB/session
    dummy_file = tmp_path / "test.wav"
    data = np.random.uniform(-0.5, 0.5, size=(44100 * 2, 2))
    sf.write(dummy_file, data, 44100)

    session_id = "test-session-id"
    save_file_to_db(session_id, str(dummy_file), "temp", "vocals" )  # mock or actual store
    mock_save.return_value = str(dummy_file)  # return a real path if needed

    intent = {
        "type": "remix",
        "instructions": {
            "volumes": {"vocals": 1.2, "drums": 1.0, "bass": 1.0, "other": 1.0},
            "reverb": {"vocals": 0.5}
        }
    }

    result = handle_remix(intent, session_id)
    assert "remix" in result
    remix_path = "separated/" + result["remix"]["file_url"].split("/")[-1]
    assert os.path.exists(remix_path)

@pytest.fixture(scope="module")
def dummy_wav(tmp_path_factory):
    # Generate a short dummy stereo wav file for testing
    tmpdir = tmp_path_factory.mktemp("data")
    path = os.path.join(tmpdir, "test.wav")
    sr = 44100
    data = np.random.uniform(-0.5, 0.5, size=(sr * 2, 2))  # 2 sec stereo noise
    sf.write(path, data, sr)
    return path

#Checks scaled values are within the expected amplitude range.
def test_apply_gain_scaling(dummy_wav):
    arr = np.random.uniform(-0.5, 0.5, size=(2, 10000))  # dummy stereo array
    stem_arrays = {"vocals": arr}
    volumes = {"vocals": 0.5}
    adjusted = apply_gain_scaling(stem_arrays, volumes, 10000)
    assert len(adjusted) == 1
    assert np.all(np.abs(adjusted[0]) <= 0.5)
    scaled = adjusted[0]
    expected = arr * 0.5
    assert np.allclose(scaled, expected[:, :10000], atol=1e-3)

def test_apply_reverb(dummy_wav):
    out = dummy_wav.replace(".wav", "_reverb.wav")
    y_in, sr = sf.read(dummy_wav)
    result = apply_reverb(dummy_wav, out, reverberance=30)
    y_out, sr_out = sf.read(out)

    assert os.path.exists(result)
    assert y_out.shape[0] > 0

    # ✅ Check that output has higher or similar RMS energy
    rms_in = np.sqrt(np.mean(y_in**2))
    rms_out = np.sqrt(np.mean(y_out**2))
    assert rms_out >= rms_in

    os.remove(result)

def test_change_pitch(dummy_wav):
    out = dummy_wav.replace(".wav", "_pitch.wav")
    y_in, sr_in = sf.read(dummy_wav)
    result = change_pitch(dummy_wav, out, n_steps=2)
    y_out, sr_out = sf.read(out)

    assert os.path.exists(result)
    assert y_out.shape[0] > 0
    assert sr_in == sr_out

    assert not np.allclose(y_in, y_out, atol=1e-3)

    os.remove(result)


def test_apply_compression(dummy_wav):
    out = dummy_wav.replace(".wav", "_comp.wav")
    result = apply_compression(dummy_wav, out)
    assert os.path.exists(result)
    y, sr = sf.read(result)
    assert y.shape[0] > 0
    os.remove(result)


def test_change_pitch_extreme(dummy_wav):
    out = dummy_wav.replace(".wav", "_pitch_extreme.wav")
    result = change_pitch(dummy_wav, out, n_steps=12)
    assert os.path.exists(result)
    y, sr = sf.read(result)
    assert y.shape[0] > 0

def test_apply_gain_scaling_exact(dummy_wav):
    y_in, sr = sf.read(dummy_wav)
    stem_arrays = {"vocals": y_in.T}  # assuming your function expects (channels, samples)
    volumes = {"vocals": 2.0}

    adjusted = apply_gain_scaling(stem_arrays, volumes, y_in.shape[0])
    y_out = adjusted[0].T  # transpose back if needed

    #Check that output is roughly input * 2.0 (within clipping limits)
    expected = y_in * 2.0
    np.testing.assert_allclose(y_out, expected, atol=1e-3)


#reverb extends decay tail energy, as expected from reverb effects
def test_apply_reverb_has_tail(dummy_wav):
    import soundfile as sf
    import librosa
    from audio_utils.remix import apply_reverb

    out = dummy_wav.replace(".wav", "_reverb.wav")
    result = apply_reverb(dummy_wav, out, reverberance=80)
    y_in, sr = sf.read(dummy_wav)
    y_out, sr_out = sf.read(result)

    assert y_in.shape == y_out.shape

    rms_in = librosa.feature.rms(y=y_in.T)[0]
    rms_out = librosa.feature.rms(y=y_out.T)[0]

    tail_in = np.mean(rms_in[int(len(rms_in)*0.9):])
    tail_out = np.mean(rms_out[int(len(rms_out)*0.9):])

    assert tail_out >= tail_in, "Expected reverb to increase or extend tail energy"

    os.remove(result)


"""
DONE:
Uses pytest fixtures to create a reusable dummy WAV file
Confirms output files are generated and readable
Uses tmp_path_factory to store temporary test files cleanly
Tests each function in isolation → fast debugging cycle

TODO: 
Add edge cases (zero-length input, invalid path)

Compare pre/post signal properties (RMS, pitch) to validate effect accuracy
"""
