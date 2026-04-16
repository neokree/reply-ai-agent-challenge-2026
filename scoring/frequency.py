# scoring/frequency.py
import pandas as pd
from datetime import datetime, timedelta

def frequency_scorer(
    sender_id: str,
    timestamp: datetime,
    all_transactions: pd.DataFrame,
    profile: dict
) -> float:
    """
    Score transaction based on velocity/frequency anomaly.

    Returns:
        Risk score 0.0-1.0
    """
    user_txs = all_transactions[all_transactions["sender_id"] == sender_id].copy()

    if user_txs.empty:
        return 0.0

    # Count transactions in last hour
    hour_ago = timestamp - timedelta(hours=1)
    txs_last_hour = len(user_txs[
        (user_txs["timestamp"] >= hour_ago) &
        (user_txs["timestamp"] <= timestamp)
    ])

    # Count transactions in last 24 hours
    day_ago = timestamp - timedelta(hours=24)
    txs_last_day = len(user_txs[
        (user_txs["timestamp"] >= day_ago) &
        (user_txs["timestamp"] <= timestamp)
    ])

    avg_daily = profile.get("avg_daily_tx_count", 2.0)

    # Scoring logic
    if txs_last_hour > 5:
        return 1.0
    elif txs_last_day > avg_daily * 3:
        return 1.0
    elif txs_last_day > avg_daily * 2:
        return 0.5
    else:
        return 0.0
