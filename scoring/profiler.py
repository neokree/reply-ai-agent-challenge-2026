import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

def build_user_profiles(
    train_transactions: pd.DataFrame,
    users: list[dict]
) -> dict:
    """
    Build behavioral profiles for each user from training data.

    Args:
        train_transactions: Training transactions DataFrame
        users: List of user dicts from users.json

    Returns:
        Dict mapping user IBAN to profile dict
    """
    # Create user lookup by IBAN
    user_by_iban = {u["iban"]: u for u in users}

    profiles = {}

    for iban, user_data in user_by_iban.items():
        # Get all transactions where user is sender
        user_txs = train_transactions[train_transactions["sender_iban"] == iban]

        if user_txs.empty:
            # Fallback for users with no training transactions
            profiles[iban] = _create_fallback_profile(user_data, train_transactions)
            continue

        # Calculate stats by transaction type
        avg_by_type = user_txs.groupby("transaction_type")["amount"].mean().to_dict()
        std_by_type = user_txs.groupby("transaction_type")["amount"].std().fillna(0).to_dict()

        # Typical hours
        hours = user_txs["timestamp"].dt.hour.tolist()
        typical_hours = list(set(hours))

        # Weekend activity (>20% of txs on weekend)
        weekend_txs = user_txs[user_txs["timestamp"].dt.weekday >= 5]
        weekend_active = len(weekend_txs) / len(user_txs) > 0.2 if len(user_txs) > 0 else False

        # Payment methods count
        payment_methods = user_txs["payment_method"].value_counts().to_dict()

        # Known recipients
        recipients = set()
        for _, tx in user_txs.iterrows():
            if tx["recipient_id"]:
                recipients.add(tx["recipient_id"])
            if tx["recipient_iban"]:
                recipients.add(tx["recipient_iban"])

        # Last tx to each recipient
        last_tx_to = {}
        for recipient in recipients:
            mask = (user_txs["recipient_id"] == recipient) | (user_txs["recipient_iban"] == recipient)
            if mask.any():
                last_tx_to[recipient] = user_txs[mask]["timestamp"].max()

        # Average daily tx count
        if len(user_txs) > 0:
            date_range = (user_txs["timestamp"].max() - user_txs["timestamp"].min()).days + 1
            avg_daily_tx_count = len(user_txs) / max(date_range, 1)
        else:
            avg_daily_tx_count = 0

        # Round amounts check (>30% are round)
        round_count = sum(1 for amt in user_txs["amount"] if amt % 100 == 0)
        often_round_amounts = round_count / len(user_txs) > 0.3 if len(user_txs) > 0 else False

        # Residence coords
        residence = user_data.get("residence", {})
        residence_coords = (
            float(residence.get("lat", 0)),
            float(residence.get("lng", 0))
        )

        profiles[iban] = {
            "user_id": user_txs["sender_id"].iloc[0] if len(user_txs) > 0 else None,
            "name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
            "avg_amount_by_type": avg_by_type,
            "std_amount_by_type": std_by_type,
            "typical_hours": typical_hours,
            "weekend_active": weekend_active,
            "payment_methods": payment_methods,
            "known_recipients": recipients,
            "last_tx_to": last_tx_to,
            "avg_daily_tx_count": avg_daily_tx_count,
            "residence_coords": residence_coords,
            "salary": user_data.get("salary", 0),
            "often_round_amounts": often_round_amounts,
        }

    return profiles


def _create_fallback_profile(user_data: dict, all_transactions: pd.DataFrame) -> dict:
    """Create a fallback profile using global stats when user has no transactions."""
    global_avg = all_transactions.groupby("transaction_type")["amount"].mean().to_dict()
    global_std = all_transactions.groupby("transaction_type")["amount"].std().fillna(0).to_dict()

    residence = user_data.get("residence", {})

    return {
        "user_id": None,
        "name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
        "avg_amount_by_type": global_avg,
        "std_amount_by_type": global_std,
        "typical_hours": list(range(8, 21)),  # 8am-8pm default
        "weekend_active": False,
        "payment_methods": {},
        "known_recipients": set(),
        "last_tx_to": {},
        "avg_daily_tx_count": 0,
        "residence_coords": (float(residence.get("lat", 0)), float(residence.get("lng", 0))),
        "salary": user_data.get("salary", 0),
        "often_round_amounts": False,
    }
