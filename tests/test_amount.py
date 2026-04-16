# tests/test_amount.py
import pytest
from scoring.amount import amount_scorer

@pytest.fixture
def profile():
    return {
        "avg_amount_by_type": {"e-commerce": 100.0, "transfer": 1000.0},
        "std_amount_by_type": {"e-commerce": 25.0, "transfer": 200.0},
    }

def test_amount_scorer_normal_amount_returns_low_score(profile):
    # Amount exactly at mean -> z-score = 0 -> score = 0
    score = amount_scorer(100.0, "e-commerce", profile)
    assert score == 0.0

def test_amount_scorer_high_amount_returns_high_score(profile):
    # Amount 4 std above mean -> z-score = 4 -> score = 1.0
    score = amount_scorer(200.0, "e-commerce", profile)  # (200-100)/25 = 4
    assert score == 1.0

def test_amount_scorer_moderate_anomaly(profile):
    # Amount 2 std above mean -> z-score = 2 -> score = 0.5
    score = amount_scorer(150.0, "e-commerce", profile)  # (150-100)/25 = 2
    assert score == 0.5

def test_amount_scorer_unknown_type_uses_global_fallback(profile):
    # Unknown type uses default values
    score = amount_scorer(500.0, "unknown_type", profile)
    assert 0.0 <= score <= 1.0

def test_amount_scorer_zero_std_returns_zero(profile):
    profile["std_amount_by_type"]["e-commerce"] = 0.0
    score = amount_scorer(150.0, "e-commerce", profile)
    assert score == 0.0
