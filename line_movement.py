# line_movement.py
from typing import List, Dict
import pandas as pd

def track_line_movement(data: List[Dict], field: str, smooth_window: int = 5) -> List[Dict]:
    """
    Track delta, momentum, and smoothed momentum for spreads or totals.

    Args:
        data: List of parsed data blocks (dictionaries)
        field: "spread" or "total"
        smooth_window: Window size for smoothed momentum

    Returns:
        Updated list of dictionaries with added keys:
        - {field}_delta
        - {field}_momentum
        - {field}_momentum_smooth
    """
    # Extract the values
    values = [row.get(field) for row in data]

    # Compute delta and momentum
    deltas = [None]  # first delta is None
    momentum = [None]  # first momentum is None

    for i in range(1, len(values)):
        if values[i] is None or values[i-1] is None:
            deltas.append(None)
            momentum.append(None)
        else:
            delta = values[i] - values[i-1]
            deltas.append(delta)
            # Momentum is sum of previous delta + current
            prev_momentum = momentum[i-1] if momentum[i-1] is not None else 0
            momentum.append(prev_momentum + delta)

    # Smoothed momentum using rolling mean
    smoothed = pd.Series(momentum).rolling(window=smooth_window, min_periods=1).mean().tolist()

    # Store in data
    for i, row in enumerate(data):
        row[f"{field}_delta"] = deltas[i]
        row[f"{field}_momentum"] = momentum[i]
        row[f"{field}_momentum_smooth"] = smoothed[i]

    return data
