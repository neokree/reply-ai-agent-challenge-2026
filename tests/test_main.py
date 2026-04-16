# tests/test_main.py
import pytest
from unittest.mock import patch, Mock
from main import create_model, generate_session_id


def test_generate_session_id_format():
    with patch.dict("os.environ", {"TEAM_NAME": "Test Team"}):
        session_id = generate_session_id()
        assert session_id.startswith("Test-Team-")
        assert len(session_id) > 15


def test_create_model_returns_chat_model():
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        model = create_model()
        assert model is not None
