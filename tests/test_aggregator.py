# tests/test_aggregator.py
import pytest
import pandas as pd
from datetime import datetime
from scoring.aggregator import ScoreAggregator

@pytest.fixture
def sample_data():
    return {
        "profiles": {
            "IBAN1": {
                "avg_amount_by_type": {"e-commerce": 100},
                "std_amount_by_type": {"e-commerce": 25},
                "typical_hours": [10, 11, 12, 14, 15],
                "weekend_active": False,
                "payment_methods": {"debit card": 10},
                "known_recipients": {"SHOP1"},
                "last_tx_to": {},
                "salary": 50000,
                "often_round_amounts": False,
            }
        },
        "eval_transactions": pd.DataFrame({
            "transaction_id": ["tx1", "tx2"],
            "sender_id": ["USER1", "USER1"],
            "sender_iban": ["IBAN1", "IBAN1"],
            "recipient_id": ["SHOP1", "NEW_SHOP"],
            "recipient_iban": ["", ""],
            "transaction_type": ["e-commerce", "e-commerce"],
            "amount": [100.0, 500.0],
            "balance_after": [900.0, 400.0],
            "payment_method": ["debit card", "PayPal"],
            "location": ["", ""],
            "timestamp": pd.to_datetime(["2087-03-15 11:00", "2087-03-15 03:00"]),
        }),
        "locations": [],
    }

def test_aggregator_returns_dataframe(sample_data):
    aggregator = ScoreAggregator(sample_data["profiles"])

    result = aggregator.score_all(
        sample_data["eval_transactions"],
        sample_data["locations"]
    )

    assert isinstance(result, pd.DataFrame)
    assert "transaction_id" in result.columns
    assert "total_risk" in result.columns

def test_aggregator_scores_in_range(sample_data):
    aggregator = ScoreAggregator(sample_data["profiles"])

    result = aggregator.score_all(
        sample_data["eval_transactions"],
        sample_data["locations"]
    )

    score_cols = [c for c in result.columns if c.endswith("_score")]
    for col in score_cols:
        assert all(0 <= result[col]) and all(result[col] <= 1)

def test_aggregator_anomalous_tx_has_higher_score(sample_data):
    aggregator = ScoreAggregator(sample_data["profiles"])

    result = aggregator.score_all(
        sample_data["eval_transactions"],
        sample_data["locations"]
    )

    # tx2 should have higher risk (unusual amount, time, new recipient, new method)
    tx1_risk = result[result["transaction_id"] == "tx1"]["total_risk"].iloc[0]
    tx2_risk = result[result["transaction_id"] == "tx2"]["total_risk"].iloc[0]

    assert tx2_risk > tx1_risk
