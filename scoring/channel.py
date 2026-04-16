# scoring/channel.py

def channel_switch_scorer(
    payment_method: str,
    amount: float,
    profile: dict
) -> float:
    """
    Score transaction based on unusual payment method.

    Returns:
        Risk score 0.0-1.0
    """
    methods = profile.get("payment_methods", {})
    salary = profile.get("salary", 50000)

    if not methods:
        return 0.0

    total_txs = sum(methods.values())
    if total_txs == 0:
        return 0.0

    method_count = methods.get(payment_method, 0)
    method_pct = method_count / total_txs

    if method_pct == 0:
        # Method NEVER used before
        score = 0.8
    elif method_pct <= 0.1:
        # Method used rarely (<=10% of txs)
        score = 0.4
    else:
        score = 0.0

    # Boost if combined with high amount
    if amount > salary * 0.5:
        score = min(score + 0.2, 1.0)

    return score
