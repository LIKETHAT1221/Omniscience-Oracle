import numpy as np
from typing import Dict, List
from scipy.stats import norm

def calculate_greeks(series: np.array, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Calculate option Greeks-inspired metrics for sports markets"""
    if len(series) < 20:
        return {}
    
    returns = np.diff(np.log(series))
    
    # Delta: Sensitivity to price changes
    delta = np.mean(returns[-5:]) if len(returns) >= 5 else 0
    
    # Gamma: Rate of change of delta
    gamma = np.std(returns[-10:]) if len(returns) >= 10 else 0
    
    # Theta: Time decay (simplified for sports markets)
    theta = -abs(np.mean(returns)) if len(returns) > 0 else 0
    
    # Vega: Volatility sensitivity
    vega = np.std(returns) if len(returns) > 0 else 0
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

def calculate_implied_volatility(series: np.array, period: int = 20) -> float:
    """Calculate implied volatility for the series"""
    if len(series) < period:
        return None
        
    returns = np.diff(np.log(series))
    return np.std(returns) * np.sqrt(252)  # Annualized

def calculate_value_at_risk(series: np.array, confidence: float = 0.95, period: int = 20) -> float:
    """Calculate Value at Risk"""
    if len(series) < period:
        return None
        
    returns = np.diff(np.log(series))
    var = np.percentile(returns, (1 - confidence) * 100)
    return var

def calculate_expected_shortfall(series: np.array, confidence: float = 0.95, period: int = 20) -> float:
    """Calculate Expected Shortfall/CVaR"""
    if len(series) < period:
        return None
        
    returns = np.diff(np.log(series))
    var = calculate_value_at_risk(series, confidence, period)
    es = np.mean(returns[returns <= var])
    return es
