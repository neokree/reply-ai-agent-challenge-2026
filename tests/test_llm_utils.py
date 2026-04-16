# tests/test_llm_utils.py
import pytest
import json
from agents.llm_utils import strip_thinking_tags, parse_json_response, parse_transaction_ids

def test_strip_thinking_tags_removes_think_block():
    response = """<think>
Let me analyze this...
The user seems suspicious because...
</think>

{"signals": []}"""

    cleaned = strip_thinking_tags(response)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert '{"signals": []}' in cleaned

def test_strip_thinking_tags_handles_no_tags():
    response = '{"signals": []}'
    cleaned = strip_thinking_tags(response)
    assert cleaned == '{"signals": []}'

def test_parse_json_response_valid_json():
    response = '{"signals": [{"user": "test", "severity": "high"}]}'
    result = parse_json_response(response)
    assert result["signals"][0]["user"] == "test"

def test_parse_json_response_extracts_json_from_text():
    response = """Here is my analysis:

    {"signals": [{"user": "test"}]}

    That's my conclusion."""
    result = parse_json_response(response)
    assert result["signals"][0]["user"] == "test"

def test_parse_json_response_with_thinking_tags():
    response = """<think>thinking...</think>
    {"signals": [{"user": "test"}]}"""
    result = parse_json_response(response)
    assert result["signals"][0]["user"] == "test"

def test_parse_json_response_returns_empty_on_failure():
    response = "No JSON here at all"
    result = parse_json_response(response)
    assert result == {"signals": []}

def test_parse_transaction_ids_valid():
    response = """<think>reasoning</think>
tx-001
tx-002
tx-003"""
    valid_ids = {"tx-001", "tx-002", "tx-003", "tx-004"}
    result = parse_transaction_ids(response, valid_ids)
    assert result == ["tx-001", "tx-002", "tx-003"]

def test_parse_transaction_ids_filters_invalid():
    response = """tx-001
invalid-id
tx-002"""
    valid_ids = {"tx-001", "tx-002", "tx-003", "tx-004"}
    result = parse_transaction_ids(response, valid_ids)
    assert "invalid-id" not in result
    assert len(result) == 2

def test_parse_transaction_ids_raises_on_empty():
    response = "no valid ids here"
    valid_ids = {"tx-001", "tx-002"}
    with pytest.raises(ValueError, match="empty"):
        parse_transaction_ids(response, valid_ids)

def test_parse_transaction_ids_raises_on_all():
    response = """tx-001
tx-002"""
    valid_ids = {"tx-001", "tx-002"}
    with pytest.raises(ValueError, match="all"):
        parse_transaction_ids(response, valid_ids)
