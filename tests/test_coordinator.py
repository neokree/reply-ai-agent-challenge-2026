# tests/test_coordinator.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from agents.coordinator import FraudCoordinator, build_coordinator_prompt

def test_build_coordinator_prompt_includes_risk_table():
    risk_df = pd.DataFrame({
        "transaction_id": ["tx1", "tx2"],
        "total_risk": [0.8, 0.2],
        "amount_score": [0.9, 0.1],
    })

    prompt = build_coordinator_prompt(
        risk_table=risk_df,
        comms_signals={"signals": []},
        eval_transactions=pd.DataFrame({
            "transaction_id": ["tx1", "tx2"],
            "amount": [1000, 50],
        })
    )

    assert "tx1" in prompt
    assert "0.8" in prompt

@patch("agents.coordinator.call_llm_with_retry")
def test_coordinator_returns_flagged_ids(mock_llm):
    mock_response = Mock()
    mock_response.content = """<think>analyzing...</think>

tx-001
tx-002"""
    mock_llm.return_value = mock_response

    coordinator = FraudCoordinator(model=Mock(), session_id="test-123")

    result = coordinator.decide(
        risk_table=pd.DataFrame({
            "transaction_id": ["tx-001", "tx-002", "tx-003"],
            "total_risk": [0.9, 0.8, 0.1],
        }),
        comms_signals={"signals": []},
        eval_transactions=pd.DataFrame({
            "transaction_id": ["tx-001", "tx-002", "tx-003"],
            "amount": [1000, 500, 50],
        })
    )

    assert "tx-001" in result
    assert "tx-002" in result
    assert "tx-003" not in result

@patch("agents.coordinator.call_llm_with_retry")
def test_coordinator_fallback_on_empty(mock_llm):
    mock_response = Mock()
    mock_response.content = "I cannot determine any fraud"
    mock_llm.return_value = mock_response

    coordinator = FraudCoordinator(model=Mock(), session_id="test-123")

    risk_df = pd.DataFrame({
        "transaction_id": [f"tx-{i}" for i in range(10)],
        "total_risk": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
    })

    result = coordinator.decide(
        risk_table=risk_df,
        comms_signals={"signals": []},
        eval_transactions=pd.DataFrame({
            "transaction_id": [f"tx-{i}" for i in range(10)],
            "amount": [100] * 10,
        })
    )

    # Should fallback to top 15% = 2 transactions
    assert len(result) >= 1
