import os
import numpy as np
import soundfile as sf
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from llm_backend.interpreter import classify_prompt, apply_feedback_to_instructions

try:
    from audio_utils.remix import (
        handle_remix, apply_gain_scaling, apply_reverb_pydub,
        change_pitch_pydub, apply_compression_pydub, generate_remix_name
    )
    REMIX_AVAILABLE = True
except ImportError:
    REMIX_AVAILABLE = False

try:
    from audio_utils.helpers import numpy_array_to_audiosegment
    HELPERS_AVAILABLE = True
except ImportError:
    HELPERS_AVAILABLE = False

@pytest.fixture
def create_test_audio():
    def _create_audio(duration=2.0, sample_rate=44100, frequency=440):
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.column_stack([
            0.3 * np.sin(2 * np.pi * frequency * t),  # Left channel
            0.3 * np.sin(2 * np.pi * frequency * t)   # Right channel
        ])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            return temp_file.name, audio_data, sample_rate
    return _create_audio

@pytest.fixture
def mock_separation_output():
    #tensors that mimic Demucs output
    sample_rate = 44100
    duration = 2.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)

    stems = {}
    for i, stem in enumerate(["vocals", "drums", "bass", "other"]):
        freq = 440 + (i * 110)  # 440, 550, 660, 770 Hz
        audio = 0.2 * np.sin(2 * np.pi * freq * t)
        stereo_audio = np.stack([audio, audio])
        stems[stem] = type('MockTensor', (), {
            'numpy': lambda: stereo_audio,
            'ndim': 2,
            'shape': stereo_audio.shape
        })()

    return stems

@pytest.fixture(scope="module")
def dummy_wav(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")
    path = os.path.join(tmpdir, "test.wav")
    sr = 44100
    data = np.random.uniform(-0.5, 0.5, size=(sr * 2, 2))  # 2 sec stereo noise
    sf.write(path, data, sr)
    return path

@pytest.mark.skipif(not REMIX_AVAILABLE, reason="Remix functions not available")
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


class TestLLMIntegration:
    @patch('llm_backend.interpreter.client')
    def test_classify_prompt_separation(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"type": "separation", "stems": ["vocals", "drums"]}'
        mock_client.chat.completions.create.return_value = mock_response

        result = classify_prompt("give me vocals and drums")

        assert result["type"] == "separation"
        assert "vocals" in result["stems"]
        assert "drums" in result["stems"]

    @patch('llm_backend.interpreter.client')
    def test_classify_prompt_remix(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '''
        {
            "type": "remix",
            "instructions": {
                "volumes": {"vocals": 1.2, "drums": 1.0, "bass": 1.0, "other": 1.0},
                "reverb": {"vocals": 0.5}
            }
        }
        '''
        mock_client.chat.completions.create.return_value = mock_response

        result = classify_prompt("make vocals louder and add reverb")

        assert result["type"] == "remix"
        assert result["instructions"]["volumes"]["vocals"] == 1.2
        assert result["instructions"]["reverb"]["vocals"] == 0.5

    @patch('llm_backend.interpreter.client')
    def test_classify_prompt_clarification(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"type": "clarification", "reason": "unclear_intent"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = classify_prompt("what can you do?")

        assert result["type"] == "clarification"
        assert result["reason"] == "unclear_intent"

class TestFeedbackSystem:
    def test_apply_feedback_to_instructions_volume(self):
        last_instructions = {
            "volumes": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}
        }

        feedback_adjustments = {
            "volumes": {"vocals": "louder", "drums": "softer"}
        }

        result = apply_feedback_to_instructions(feedback_adjustments, last_instructions)

        assert result["volumes"]["vocals"] > 1.0
        assert result["volumes"]["drums"] < 1.0
        assert result["volumes"]["bass"] == 1.0

    def test_apply_feedback_to_instructions_pitch(self):
        last_instructions = {
            "volumes": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0},
            "pitch_shift": {"vocals": 0}
        }

        feedback_adjustments = {
            "pitch_shift": {"vocals": "-4"}
        }

        result = apply_feedback_to_instructions(feedback_adjustments, last_instructions)

        assert result["pitch_shift"]["vocals"] == -4

    def test_generate_remix_name(self):
        intent = {
            "instructions": {
                "volumes": {"vocals": 1.2},
                "reverb": {"vocals": 0.5},
                "pitch_shift": {"vocals": 2}
            }
        }

        name = generate_remix_name(intent)

        assert name.startswith("remix_")
        assert name.endswith(".wav")
        assert "reverb" in name
        assert "pitch" in name
