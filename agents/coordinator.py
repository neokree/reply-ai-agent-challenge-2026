# agents/coordinator.py
import logging
import pandas as pd
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler

from .llm_utils import parse_transaction_ids, call_llm_with_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the fraud coordinator for MirrorPay's anti-fraud system.

You receive:
1. A risk score table computed by algorithmic analyzers (scores 0-1 per category)
2. Social engineering signals detected in communications
3. The transactions to evaluate

RULES:
- You MUST flag at least 1 transaction and CANNOT flag all transactions
- Missing fraud causes severe financial damage — catching fraud is the primary goal
- Expect roughly 10-20% of transactions to be fraudulent (8-16 out of 80)
- Flag ANY transaction with total_risk >= 0.15 unless there is strong evidence it is legitimate
- Flag ALL transactions linked to users who appear in the communication signals with high or medium severity
- Correlate signals: high algorithmic score + communication signal = almost certainly fraud
- Temporal correlation: suspicious communication within 24h of an anomalous transaction = flag it
- When in doubt, flag — a false positive is far less damaging than missing real fraud

Respond ONLY with the list of transaction_id to flag as fraudulent, one per line.
No other text, no explanations."""


def build_coordinator_prompt(
    risk_table: pd.DataFrame,
    comms_signals: dict[str, Any],
    eval_transactions: pd.DataFrame
) -> str:
    """Build the user prompt for the coordinator."""
    parts = []

    # Risk score table
    parts.append("## RISK SCORE TABLE")
    parts.append(risk_table.to_string(index=False))
    parts.append("")

    # Communications signals
    parts.append("## COMMUNICATION SIGNALS")
    signals = comms_signals.get("signals", [])
    if signals:
        for sig in signals:
            parts.append(f"- User {sig.get('user_iban', 'unknown')}: {sig.get('severity', 'unknown')} - {sig.get('reason', 'no reason')}")
    else:
        parts.append("No suspicious communication signals detected.")
    parts.append("")

    # Transaction details
    parts.append("## TRANSACTIONS TO EVALUATE")
    cols = ["transaction_id", "sender_id", "recipient_id", "amount",
            "transaction_type", "timestamp"]
    available_cols = [c for c in cols if c in eval_transactions.columns]
    parts.append(eval_transactions[available_cols].to_string(index=False))

    return "\n".join(parts)


class FraudCoordinator:
    """Coordinator agent for final fraud decision."""

    def __init__(self, model, session_id: str):
        self.model = model
        self.session_id = session_id

    def decide(
        self,
        risk_table: pd.DataFrame,
        comms_signals: dict[str, Any],
        eval_transactions: pd.DataFrame
    ) -> list[str]:
        """
        Make final fraud decision.

        Returns:
            List of transaction IDs flagged as fraudulent
        """
        try:
            handler = CallbackHandler()
        except Exception:
            handler = None

        user_prompt = build_coordinator_prompt(
            risk_table, comms_signals, eval_transactions
        )

        valid_ids = set(eval_transactions["transaction_id"].tolist())

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        config = {"metadata": {"langfuse_session_id": self.session_id}}
        if handler:
            config["callbacks"] = [handler]

        try:
            response = call_llm_with_retry(
                self.model,
                messages,
                config=config
            )

            return parse_transaction_ids(response.content, valid_ids)

        except ValueError as e:
            # Fallback to algorithmic decision
            logger.warning(f"Coordinator fallback triggered: {e}")
            return self._algorithmic_fallback(risk_table, len(valid_ids))

        except Exception as e:
            logger.error(f"Coordinator failed: {e}")
            return self._algorithmic_fallback(risk_table, len(valid_ids))

    def _algorithmic_fallback(
        self,
        risk_table: pd.DataFrame,
        total_count: int
    ) -> list[str]:
        """Fallback to algorithmic decision based on risk scores."""
        sorted_df = risk_table.sort_values("total_risk", ascending=False)

        # Flag top 20%, minimum 1, maximum 35%
        n_to_flag = max(1, min(int(total_count * 0.20), int(total_count * 0.35)))

        flagged = sorted_df.head(n_to_flag)["transaction_id"].tolist()

        logger.info(f"Algorithmic fallback: flagging {len(flagged)} transactions")
        return flagged
