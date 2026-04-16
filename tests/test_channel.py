# tests/test_channel.py
import pytest
from scoring.channel import channel_switch_scorer

@pytest.fixture
def profile():
    return {
        "payment_methods": {"debit card": 45, "mobile device": 5},
        "salary": 50000,
    }

def test_channel_usual_method_returns_zero(profile):
    score = channel_switch_scorer("debit card", 100.0, profile)
    assert score == 0.0

def test_channel_rare_method_returns_moderate(profile):
    # mobile device is 10% of txs
    score = channel_switch_scorer("mobile device", 100.0, profile)
    assert score == 0.4

def test_channel_new_method_returns_high(profile):
    score = channel_switch_scorer("PayPal", 100.0, profile)
    assert score == 0.8

def test_channel_new_method_high_amount_boost(profile):
    # New method + amount > 50% of salary
    score = channel_switch_scorer("PayPal", 30000.0, profile)
    assert score == 1.0
