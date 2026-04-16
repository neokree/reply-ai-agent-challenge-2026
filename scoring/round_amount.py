# scoring/round_amount.py

def round_amount_scorer(amount: float, profile: dict) -> float:
    """
    Score transaction based on suspiciously round amount.

    Returns:
        Risk score 0.0-1.0
    """
    is_round_100 = (amount % 100 == 0) and (amount >= 100)
    is_round_1000 = (amount % 1000 == 0) and (amount >= 1000)

    if is_round_1000:
        score = 0.5
    elif is_round_100:
        score = 0.3
    else:
        score = 0.0

    # Reduce penalty if user often has round amounts (e.g., fixed rent)
    if profile.get("often_round_amounts", False):
        score *= 0.3

    return score
