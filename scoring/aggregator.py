# scoring/aggregator.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from .amount import amount_scorer
from .time import time_scorer
from .frequency import frequency_scorer
from .channel import channel_switch_scorer
from .round_amount import round_amount_scorer
from .entities import new_entity_scorer
from .balance import balance_drain_scorer
from .geo import geo_scorer


class ScoreAggregator:
    """Aggregate all rule-based scores for transactions."""

    # Default weights for each scorer
    WEIGHTS = {
        "amount_score": 0.15,
        "time_score": 0.08,
        "geo_score": 0.15,
        "new_entity_score": 0.12,
        "frequency_score": 0.08,
        "channel_switch_score": 0.07,
        "round_amount_score": 0.05,
        "balance_drain_score": 0.10,
        "isolation_forest_score": 0.10,
        "graph_score": 0.10,
    }

    def __init__(self, profiles: dict):
        self.profiles = profiles

    def score_all(
        self,
        eval_transactions: pd.DataFrame,
        locations: list[dict],
        ml_scores: Optional[dict[str, float]] = None,
        graph_scores: Optional[dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Score all transactions.

        Args:
            eval_transactions: Evaluation transactions DataFrame
            locations: Location records for geo checking
            ml_scores: Optional isolation forest scores
            graph_scores: Optional graph analysis scores

        Returns:
            DataFrame with all scores and total_risk
        """
        ml_scores = ml_scores or {}
        graph_scores = graph_scores or {}

        results = []

        # Sort by timestamp for prev_tx tracking
        sorted_txs = eval_transactions.sort_values("timestamp").reset_index(drop=True)

        # Track previous tx per user for geo velocity check
        prev_tx_by_user: dict[str, dict] = {}

        for idx, tx in sorted_txs.iterrows():
            tx_id = tx["transaction_id"]
            sender_iban = tx["sender_iban"] if "sender_iban" in tx.index else ""
            sender_id = tx["sender_id"] if "sender_id" in tx.index else ""

            # Get profile (fallback to empty)
            profile = self.profiles.get(sender_iban, {})

            # Get previous tx for this user
            prev_tx = prev_tx_by_user.get(sender_id)

            # Get optional fields safely
            tx_location = tx["location"] if "location" in tx.index else None
            tx_payment_method = tx["payment_method"] if "payment_method" in tx.index else ""
            tx_transaction_type = tx["transaction_type"] if "transaction_type" in tx.index else ""
            tx_recipient_id = tx["recipient_id"] if "recipient_id" in tx.index else None
            tx_recipient_iban = tx["recipient_iban"] if "recipient_iban" in tx.index else None
            tx_balance_after = tx["balance_after"] if "balance_after" in tx.index else 0

            # Determine recipient for entity scoring
            recipient = tx_recipient_id or tx_recipient_iban

            # Calculate all scores
            scores = {
                "transaction_id": tx_id,
                "amount_score": amount_scorer(
                    tx["amount"],
                    tx_transaction_type,
                    profile
                ),
                "time_score": time_scorer(
                    tx["timestamp"],
                    profile
                ),
                "geo_score": geo_scorer(
                    tx_location,
                    tx["timestamp"],
                    sender_id,
                    locations,
                    prev_tx.get("location") if prev_tx else None,
                    prev_tx.get("timestamp") if prev_tx else None
                ),
                "new_entity_score": new_entity_scorer(
                    recipient,
                    tx["timestamp"],
                    tx["amount"],
                    profile
                ),
                "frequency_score": frequency_scorer(
                    sender_id,
                    tx["timestamp"],
                    sorted_txs,
                    profile
                ),
                "channel_switch_score": channel_switch_scorer(
                    tx_payment_method,
                    tx["amount"],
                    profile
                ),
                "round_amount_score": round_amount_scorer(
                    tx["amount"],
                    profile
                ),
                "balance_drain_score": balance_drain_scorer(
                    tx["amount"],
                    tx_balance_after,
                    sender_id,
                    tx["timestamp"],
                    sorted_txs
                ),
                "isolation_forest_score": ml_scores.get(tx_id, 0.0),
                "graph_score": graph_scores.get(tx_id, 0.0),
            }

            # Calculate weighted total
            total = sum(
                scores[k] * self.WEIGHTS.get(k, 0)
                for k in self.WEIGHTS.keys()
                if k in scores
            )
            scores["total_risk"] = total

            results.append(scores)

            # Update prev tx for velocity check
            if tx_location:
                prev_tx_by_user[sender_id] = {
                    "location": tx_location,
                    "timestamp": tx["timestamp"]
                }

        return pd.DataFrame(results)
