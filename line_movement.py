# line_movement.py
from typing import List, Dict, Optional

def calculate_delta(current: float, previous: float) -> float:
    """
    Calculate the raw delta between two spreads/totals.
    Negative values preserved correctly.
    Example:
      -3.5 → -2.5 = +1.0 (line moving toward underdog)
      45.5 → 46.5 = +1.0 (line moving toward the over)
    """
    if current is None or previous is None:
        return 0.0
    return current - previous


def calculate_momentum(history: List[float]) -> float:
    """
    Momentum = average directional delta across history.
    Positive = moving toward underdog/over
    Negative = moving toward favorite/under
    """
    if not history or len(history) < 2:
        return 0.0
    deltas = [calculate_delta(history[i], history[i-1]) for i in range(1, len(history))]
    return sum(deltas) / len(deltas)


def track_line_movement(data: List[Dict], key: str) -> List[Dict]:
    """
    Annotate parsed odds data with delta & momentum for spreads/totals.
    Args:
      data: List of parsed odds rows (from parse_blocks_strict)
      key: 'spread' or 'total'
    Returns:
      Same list, with 'delta' and 'momentum' fields added.
    """
    history = []
    for row in data:
        val = row.get(key)
        if val is None:
            row[f"{key}_delta"] = 0.0
            row[f"{key}_momentum"] = 0.0
        else:
            prev = history[-1] if history else None
            delta = calculate_delta(val, prev) if prev is not None else 0.0
            history.append(val)
            row[f"{key}_delta"] = delta
            row[f"{key}_momentum"] = calculate_momentum(history)
    return data
