# tests/test_frequency.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from scoring.frequency import frequency_scorer

@pytest.fixture
def profile():
    return {"avg_daily_tx_count": 2.0}

def test_frequency_scorer_normal_returns_zero(profile):
    current_ts = datetime(2087, 1, 10, 14, 0, 0)
    all_txs = pd.DataFrame({
        "sender_id": ["USER1", "USER1"],
        "timestamp": [
            datetime(2087, 1, 10, 13, 30, 0),
            current_ts,
        ]
    })
    score = frequency_scorer("USER1", current_ts, all_txs, profile)
    assert score == 0.0

def test_frequency_scorer_burst_returns_high(profile):
    current_ts = datetime(2087, 1, 10, 14, 0, 0)
    timestamps = [current_ts - timedelta(minutes=i*10) for i in range(6)]
    all_txs = pd.DataFrame({
        "sender_id": ["USER1"] * 6,
        "timestamp": timestamps,
    })
    score = frequency_scorer("USER1", current_ts, all_txs, profile)
    assert score == 1.0

def test_frequency_scorer_high_daily_returns_moderate(profile):
    current_ts = datetime(2087, 1, 10, 14, 0, 0)
    timestamps = [current_ts - timedelta(hours=i*3) for i in range(7)]
    all_txs = pd.DataFrame({
        "sender_id": ["USER1"] * 7,
        "timestamp": timestamps,
    })
    score = frequency_scorer("USER1", current_ts, all_txs, profile)
    assert score >= 0.5
