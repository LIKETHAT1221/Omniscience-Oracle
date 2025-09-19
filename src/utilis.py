import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

def validate_game_data(game_data: Dict) -> bool:
    required_fields = ['game_type', 'datetime']
    
    if game_data['game_type'] == 'spread':
        required_fields.extend(['favorite_team', 'spread', 'spread_vig', 'total', 'total_vig', 'away_ml', 'home_ml'])
    else:
        required_fields.extend(['away_ml', 'home_ml', 'total', 'total_vig', 'runline', 'runline_vig'])
    
    return all(field in game_data and game_data[field] is not None for field in required_fields)

def calculate_implied_probability(odds: Any) -> float:
    if odds is None:
        return None
        
    if isinstance(odds, str):
        odds = odds.strip().lower()
        if odds == 'even':
            return 0.5
        try:
            odds = float(odds)
        except ValueError:
            return None
    
    if isinstance(odds, (int, float)):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    return None

def format_recommendation(recommendation: Dict) -> str:
    return (
        f"{recommendation['recommendation']} "
        f"(Confidence: {recommendation['confidence']:.0%}, "
        f"EV: {recommendation['expected_value']:+.2%}, "
        f"Stake: {recommendation['kelly_stake']:.1%})"
    )

def safe_float_conversion(value: Any, default: float = None) -> float:
    if value is None:
        return default
        
    if isinstance(value, (int, float)):
        return float(value)
        
    if isinstance(value, str):
        try:
            clean_value = ''.join(c for c in value if c.isdigit() or c in ['-', '.', '+'])
            if clean_value:
                return float(clean_value)
        except ValueError:
            pass
            
    return default
