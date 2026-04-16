# tests/test_balance.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from scoring.balance import balance_drain_scorer

def test_balance_drain_small_pct_returns_zero():
    # 10% of balance spent
    score = balance_drain_scorer(
        amount=100.0,
        balance_after=900.0,
        sender_id="USER1",
        timestamp=datetime(2087, 3, 15, 10, 0),
        all_transactions=pd.DataFrame(columns=["sender_id", "timestamp", "amount"])
    )
    assert score == 0.0

def test_balance_drain_high_pct_returns_high():
    # 90% of balance spent -> score = 1.0
    score = balance_drain_scorer(
        amount=900.0,
        balance_after=100.0,
        sender_id="USER1",
        timestamp=datetime(2087, 3, 15, 10, 0),
        all_transactions=pd.DataFrame(columns=["sender_id", "timestamp", "amount"])
    )
    assert score == 1.0

def test_balance_drain_moderate_pct_returns_moderate():
    # 60% of balance spent
    score = balance_drain_scorer(
        amount=600.0,
        balance_after=400.0,
        sender_id="USER1",
        timestamp=datetime(2087, 3, 15, 10, 0),
        all_transactions=pd.DataFrame(columns=["sender_id", "timestamp", "amount"])
    )
    assert score == 0.4

def test_balance_drain_pattern_detected():
    # Multiple txs draining account in 2 hours
    current_ts = datetime(2087, 3, 15, 10, 0)
    all_txs = pd.DataFrame({
        "sender_id": ["USER1", "USER1", "USER1"],
        "timestamp": [
            current_ts - timedelta(hours=1),
            current_ts - timedelta(minutes=30),
            current_ts,
        ],
        "amount": [300.0, 300.0, 300.0],
    })
    # Total 900 out of 1000 balance
    score = balance_drain_scorer(
        amount=300.0,
        balance_after=100.0,
        sender_id="USER1",
        timestamp=current_ts,
        all_transactions=all_txs
    )
    assert score == 1.0
