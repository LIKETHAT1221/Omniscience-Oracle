import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import scipy.stats as stats

def calculate_momentum_indicators(series: np.array, period: int = 2) -> Dict[str, float]:
    """Calculate Momentum Velocity and Acceleration"""
    if len(series) < period + 1:
        return {'MOM_V': None, 'MOM_A': None}
    
    mom_v = (series[-1] - series[-period-1]) / period
    
    if len(series) >= 2 * period + 1:
        prev_mom_v = (series[-period-1] - series[-2*period-1]) / period
        mom_a = (mom_v - prev_mom_v) / period
    else:
        mom_a = None
        
    return {'MOM_V': mom_v, 'MOM_A': mom_a}

def calculate_rsi(series: np.array, period: int = 14) -> float:
    """Relative Strength Index"""
    if len(series) < period + 1:
        return None
        
    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.mean(gains[-period:])
    avg_losses = np.mean(losses[-period:])
    
    if avg_losses == 0:
        return 100
        
    rs = avg_gains / avg_losses
    return 100 - (100 / (1 + rs))

def calculate_z_score(series: np.array, lookback: int = 20) -> float:
    """Z-score for statistical significance"""
    if len(series) < lookback:
        return None
        
    current_value = series[-1]
    mean_val = np.mean(series[-lookback:])
    std_val = np.std(series[-lookback:])
    
    if std_val == 0:
        return 0
        
    return (current_value - mean_val) / std_val

def fibonacci_retracement(series: np.array, lookback: int = 50) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels"""
    if len(series) < lookback:
        return {}
        
    high = np.max(series[-lookback:])
    low = np.min(series[-lookback:])
    current = series[-1]
    
    fib_levels = {
        '0.0': high,
        '0.236': high - 0.236 * (high - low),
        '0.382': high - 0.382 * (high - low),
        '0.5': high - 0.5 * (high - low),
        '0.618': high - 0.618 * (high - low),
        '0.786': high - 0.786 * (high - low),
        '1.0': low,
        'current_position': (high - current) / (high - low) if high != low else 0.5
    }
    
    return fib_levels

def fibonacci_extensions(series: np.array, lookback: int = 50) -> Dict[str, float]:
    """Calculate Fibonacci extension levels"""
    if len(series) < lookback:
        return {}
        
    high = np.max(series[-lookback:])
    low = np.min(series[-lookback:])
    current = series[-1]
    
    fib_extensions = {
        '1.272': high + 0.272 * (high - low),
        '1.414': high + 0.414 * (high - low),
        '1.618': high + 0.618 * (high - low),
        '2.0': high + 1.0 * (high - low),
        '2.24': high + 1.24 * (high - low),
        '2.618': high + 1.618 * (high - low)
    }
    
    return fib_extensions

def detect_steam_movement(series: np.array, threshold: float = 2.5) -> Dict[str, bool]:
    """Detect steam movement using multiple criteria"""
    if len(series) < 10:
        return {'steam_detected': False, 'confidence': 0}
    
    # Multiple steam detection criteria
    z_score = calculate_z_score(series, 20) or 0
    mom_v = calculate_momentum_indicators(series, 3)['MOM_V'] or 0
    volatility = np.std(series[-10:]) / np.mean(series[-10:]) if np.mean(series[-10:]) != 0 else 0
    
    steam_detected = (
        abs(z_score) > threshold or 
        abs(mom_v) > np.std(series[-20:]) * 2 or
        volatility > 0.1
    )
    
    confidence = min(1.0, (abs(z_score) / threshold + abs(mom_v) / (np.std(series[-20:]) * 2) + volatility) / 3)
    
    return {'steam_detected': bool(steam_detected), 'confidence': confidence}

def adaptive_moving_average(series: np.array, base_period: int = 10, 
                           max_period: int = 30, sensitivity: float = 2.0) -> float:
    """Your Adaptive MA for finding true value"""
    if len(series) < base_period:
        return np.mean(series) if len(series) > 0 else None
    
    volatility = np.std(series[-base_period:])
    if volatility == 0:
        efficiency_ratio = 0
    else:
        direction = np.abs(series[-1] - series[-base_period])
        efficiency_ratio = direction / (volatility * np.sqrt(base_period))
    
    adaptive_period = base_period + int((max_period - base_period) * efficiency_ratio * sensitivity)
    adaptive_period = min(max(adaptive_period, base_period), max_period)
    
    return np.mean(series[-adaptive_period:])

def calculate_all_ta_indicators(series_data: List[Dict], field: str) -> Dict:
    """Calculate complete TA analysis for a market"""
    if not series_data or len(series_data) < 10:
        return {}
    
    values = [dp[field] for dp in series_data if field in dp and dp[field] is not None]
    
    if len(values) < 10:
        return {}
    
    values_array = np.array(values)
    
    return {
        'momentum': calculate_momentum_indicators(values_array),
        'rsi': calculate_rsi(values_array),
        'z_score': calculate_z_score(values_array),
        'fib_retracement': fibonacci_retracement(values_array),
        'fib_extensions': fibonacci_extensions(values_array),
        'steam_detection': detect_steam_movement(values_array),
        'adaptive_ma': adaptive_moving_average(values_array),
        'current_value': values_array[-1],
        'data_points': len(values_array)
    }
