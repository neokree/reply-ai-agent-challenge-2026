# scoring/time.py
from datetime import datetime

def time_scorer(timestamp: datetime, profile: dict) -> float:
    """
    Score transaction based on time anomaly.

    Returns:
        Risk score 0.0-1.0
    """
    tx_hour = timestamp.hour
    tx_weekday = timestamp.weekday()  # 0=Mon, 6=Sun
    is_weekend = tx_weekday >= 5

    typical_hours = profile.get("typical_hours", list(range(8, 21)))
    weekend_active = profile.get("weekend_active", False)

    # Score for unusual hour
    if tx_hour in typical_hours:
        hour_score = 0.0
    else:
        if not typical_hours:
            hour_score = 0.0
        else:
            min_distance = min(abs(tx_hour - h) for h in typical_hours)
            hour_score = min(min_distance / 6.0, 1.0)

    # Weekend boost if user doesn't normally operate on weekends
    weekend_boost = 0.3 if (is_weekend and not weekend_active) else 0.0

    return min(hour_score + weekend_boost, 1.0)
