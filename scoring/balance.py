# scoring/balance.py
import pandas as pd
from datetime import datetime, timedelta

def balance_drain_scorer(
    amount: float,
    balance_after: float,
    sender_id: str,
    timestamp: datetime,
    all_transactions: pd.DataFrame
) -> float:
    """
    Score transaction based on balance drain pattern.

    Returns:
        Risk score 0.0-1.0
    """
    # Calculate balance before
    balance_before = balance_after + amount

    if balance_before <= 0:
        return 0.0

    pct_spent = amount / balance_before

    # Single transaction scoring
    if pct_spent >= 0.9:
        score = 1.0
    elif pct_spent > 0.7:
        score = 0.7
    elif pct_spent > 0.5:
        score = 0.4
    else:
        score = 0.0

    # Check for drain pattern: multiple txs eroding balance in 2 hours
    if not all_transactions.empty:
        two_hours_ago = timestamp - timedelta(hours=2)
        recent_txs = all_transactions[
            (all_transactions["sender_id"] == sender_id) &
            (all_transactions["timestamp"] >= two_hours_ago) &
            (all_transactions["timestamp"] <= timestamp)
        ]

        if len(recent_txs) >= 3:
            total_spent = recent_txs["amount"].sum()
            if balance_before > 0 and total_spent / balance_before > 0.8:
                score = 1.0

    return score
