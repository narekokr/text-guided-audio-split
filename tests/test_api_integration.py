import pytest
from unittest.mock import patch, MagicMock
import torch

try:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    client = None

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI test dependencies not available")

@pytest.mark.parametrize(
    "request_data,expected_status,expected_keys",
    [
        (
            {"message": "test message", "session_id": "test_session", "user_id": "test_user"},
            200,
            ["reply", "history"]
        ),
        (
            {"message": "give me vocals", "session_id": "test_session", "user_id": "test_user"},
            200,
            ["reply", "history"]
        ),
        (
            {"session_id": "test_session", "user_id": "test_user"},
            422,
            None
        ),
        (
            {"message": "test message", "session_id": "test_session"},
            422,
            None
        ),
    ],
    ids=[
        "valid_basic_request",
        "valid_vocals_request",
        "missing_message",
        "missing_session_id",
    ]
)

def test_chat_endpoint_validation(request_data, expected_status, expected_keys):
    with patch('api.main.classify_prompt') as mock_classify, \
         patch('api.main.generate_clarification_response') as mock_clarify, \
         patch('api.main.get_or_create_session') as mock_session, \
         patch('api.main.save_message') as mock_save, \
         patch('api.main.get_history') as mock_history:

        mock_classify.return_value = {"type": "clarification", "reason": "test"}
        mock_clarify.return_value = "Test response"
        mock_session.return_value = None
        mock_save.return_value = None
        mock_history.return_value = {"messages": [], "audio_path": None}

        response = client.post("/chat", json=request_data)

        assert response.status_code == expected_status

        if expected_status == 200:
            data = response.json()
            for key in expected_keys:
                assert key in data

@patch('api.main.classify_prompt')
@patch('api.main.separate_audio')
@patch('api.main.get_file_from_db')
@patch('api.main.save_file_to_db')
@patch('api.main.get_or_create_session')
@patch('api.main.save_message')
@patch('api.main.get_history')
def test_separation_flow(
    mock_history, mock_save_msg, mock_session, mock_save_file,
    mock_get_file, mock_separate, mock_classify
):
    mock_classify.return_value = {"type": "separation", "stems": ["vocals", "drums"]}
    mock_session.return_value = None
    mock_save_msg.return_value = None
    mock_history.return_value = {"messages": [], "audio_path": None}
    mock_get_file.return_value = "dummy/path/audio.wav"
    mock_separate.return_value = {
        "vocals": torch.zeros((2, 1000)),
        "drums": torch.zeros((2, 1000))
    }

    response = client.post("/chat", json={
        "message": "extract vocals and drums",
        "session_id": "test_session",
        "user_id": "test_user"
    })

    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "stems" in data
    assert any(s["name"] == "vocals" for s in data["stems"])
    assert any(s["name"] == "drums" for s in data["stems"])
