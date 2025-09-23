# odds_utils.py
from sportsbetting import convert_odds, implied_probability
from typing import Tuple

def american_to_prob(odds: float) -> float:
    """
    Convert American odds to implied probability (including vig if given as raw odds).
    Args:
        odds: American odds (e.g., -110, +150)
    Returns:
        Implied probability as a float between 0 and 1
    """
    return implied_probability(odds)

def no_vig_prob(odds_a: float, odds_b: float) -> Tuple[float, float]:
    """
    Calculate no-vig (fair) probabilities for two opposing odds.
    Args:
        odds_a: American odds for side A
        odds_b: American odds for side B
    Returns:
        Tuple of (prob_a, prob_b) normalized to sum to 1
    """
    p_a = implied_probability(odds_a)
    p_b = implied_probability(odds_b)
    total = p_a + p_b
    if total == 0:
        return 0.5, 0.5
    return p_a / total, p_b / total

def american_to_decimal(odds: float) -> float:
    """
    Convert American odds to Decimal odds.
    Args:
        odds: American odds (e.g., -110, +150)
    Returns:
        Decimal odds (e.g., 1.91, 2.50)
    """
    return convert_odds(odds, to="decimal")

def decimal_to_american(decimal_odds: float) -> float:
    """
    Convert Decimal odds to American odds.
    Args:
        decimal_odds: Decimal odds (e.g., 1.91, 2.50)
    Returns:
        American odds (e.g., -110, +150)
    """
    return convert_odds(decimal_odds, to="american")
