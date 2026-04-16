# scoring/entities.py
from datetime import datetime

def new_entity_scorer(
    recipient: str,
    timestamp: datetime,
    amount: float,
    profile: dict
) -> float:
    """
    Score transaction based on whether recipient is new/unknown.

    Returns:
        Risk score 0.0-1.0
    """
    if not recipient:
        return 0.0

    known = profile.get("known_recipients", set())
    last_tx_to = profile.get("last_tx_to", {})
    salary = profile.get("salary", 50000)

    if recipient in known:
        # Known recipient - check how long since last tx
        last_date = last_tx_to.get(recipient)
        if last_date:
            days_since = (timestamp - last_date).days
            if days_since > 90:
                return 0.2
        return 0.0
    else:
        # New recipient - base score + amount factor
        amount_factor = min(amount / salary, 1.0) if salary > 0 else 0.5
        score = 0.5 + (0.5 * amount_factor)
        return score
