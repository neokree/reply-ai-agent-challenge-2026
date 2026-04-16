# tests/test_comms_agent.py
import pytest
from unittest.mock import Mock, patch
from agents.comms import CommsAgent, build_comms_prompt

def test_build_comms_prompt_includes_all_data():
    prompt = build_comms_prompt(
        sms=[{"sms": "Test SMS"}],
        mails=[{"mail": "Test Email"}],
        audio_transcripts={"file.mp3": {"user": "test", "transcript": "Hello"}},
        user_mapping={"John Doe": "IBAN123"}
    )

    assert "Test SMS" in prompt
    assert "Test Email" in prompt
    assert "Hello" in prompt
    assert "John Doe" in prompt
    assert "IBAN123" in prompt

def test_build_comms_prompt_handles_empty_audio():
    prompt = build_comms_prompt(
        sms=[{"sms": "Test"}],
        mails=[],
        audio_transcripts={},
        user_mapping={}
    )
    assert "No audio" in prompt or "Test" in prompt

@patch("agents.comms.call_llm_with_retry")
def test_comms_agent_returns_signals(mock_llm):
    mock_response = Mock()
    mock_response.content = '{"signals": [{"user_iban": "IBAN1", "severity": "high", "reason": "phishing"}]}'
    mock_llm.return_value = mock_response

    agent = CommsAgent(model=Mock(), session_id="test-123")
    result = agent.analyze(
        sms=[],
        mails=[],
        audio_transcripts={},
        user_mapping={}
    )

    assert "signals" in result
    assert len(result["signals"]) == 1

@patch("agents.comms.call_llm_with_retry")
def test_comms_agent_handles_thinking_tags(mock_llm):
    mock_response = Mock()
    mock_response.content = """<think>
Let me analyze...
</think>

{"signals": []}"""
    mock_llm.return_value = mock_response

    agent = CommsAgent(model=Mock(), session_id="test-123")
    result = agent.analyze([], [], {}, {})

    assert result == {"signals": []}
