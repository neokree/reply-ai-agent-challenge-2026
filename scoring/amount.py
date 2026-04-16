# scoring/amount.py

# Global fallback values
GLOBAL_MEAN = 500.0
GLOBAL_STD = 300.0

def amount_scorer(
    amount: float,
    transaction_type: str,
    profile: dict
) -> float:
    """
    Score transaction amount anomaly using Z-score.

    Args:
        amount: Transaction amount
        transaction_type: Type of transaction (e-commerce, transfer, etc.)
        profile: User profile dict

    Returns:
        Risk score 0.0-1.0 (higher = more anomalous)
    """
    avg_by_type = profile.get("avg_amount_by_type", {})
    std_by_type = profile.get("std_amount_by_type", {})

    mean = avg_by_type.get(transaction_type, GLOBAL_MEAN)
    std = std_by_type.get(transaction_type, GLOBAL_STD)

    if std == 0:
        return 0.0

    z_score = (amount - mean) / std
    score = min(abs(z_score) / 4.0, 1.0)

    return score
