# tests/test_round_amount.py
import pytest
from scoring.round_amount import round_amount_scorer

@pytest.fixture
def profile():
    return {"often_round_amounts": False}

def test_round_amount_non_round_returns_zero(profile):
    score = round_amount_scorer(123.45, profile)
    assert score == 0.0

def test_round_amount_100_returns_low(profile):
    score = round_amount_scorer(200.0, profile)
    assert score == 0.3

def test_round_amount_1000_returns_higher(profile):
    score = round_amount_scorer(1000.0, profile)
    assert score == 0.5

def test_round_amount_user_often_round_reduces_score(profile):
    profile["often_round_amounts"] = True
    score = round_amount_scorer(1000.0, profile)
    assert score == 0.15  # 0.5 * 0.3

def test_round_amount_small_round_ignored(profile):
    # Amount < 100 not flagged even if round
    score = round_amount_scorer(50.0, profile)
    assert score == 0.0
