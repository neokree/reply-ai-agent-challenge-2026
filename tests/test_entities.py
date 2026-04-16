# tests/test_entities.py
import pytest
from datetime import datetime, timedelta
from scoring.entities import new_entity_scorer

@pytest.fixture
def profile():
    now = datetime(2087, 3, 15, 10, 0, 0)
    return {
        "known_recipients": {"IBAN1", "MERCHANT1"},
        "last_tx_to": {
            "IBAN1": now - timedelta(days=30),
            "MERCHANT1": now - timedelta(days=100),
        },
        "salary": 50000,
    }

def test_known_recent_recipient_returns_zero(profile):
    ts = datetime(2087, 3, 15, 10, 0, 0)
    score = new_entity_scorer("IBAN1", ts, 100.0, profile)
    assert score == 0.0

def test_known_old_recipient_returns_low(profile):
    # MERCHANT1 not seen in 100 days (>90)
    ts = datetime(2087, 3, 15, 10, 0, 0)
    score = new_entity_scorer("MERCHANT1", ts, 100.0, profile)
    assert score == 0.2

def test_new_recipient_returns_high(profile):
    ts = datetime(2087, 3, 15, 10, 0, 0)
    score = new_entity_scorer("NEW_IBAN", ts, 100.0, profile)
    assert score >= 0.5

def test_new_recipient_high_amount_returns_max(profile):
    ts = datetime(2087, 3, 15, 10, 0, 0)
    # Amount = salary -> factor = 1.0 -> score = 0.5 + 0.5 = 1.0
    score = new_entity_scorer("NEW_IBAN", ts, 50000.0, profile)
    assert score == 1.0
