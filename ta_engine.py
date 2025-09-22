import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import math

class TechnicalAnalysisEngine:
    """Complete TA engine with all indicators"""
    
    def __init__(self):
        pass
    
    def analyze_all(self, parsed_data: List[Dict]) -> Dict[str, Any]:
        """Analyze all technical indicators for all data points"""
        # Extract series for analysis
        spreads = [r['spread'] for r in parsed_data if r['spread'] is not None]
        totals = [r['total'] for r in parsed_data if r['total'] is not None]
        ml_away_series = [r['ml_away'] for r in parsed_data if r['ml_away'] is not None]
        ml_home_series = [r['ml_home'] for r in parsed_data if r['ml_home'] is not None]
        
        # Process Spread
        spread_results = self._analyze_spread(spreads) if spreads else {}
        
        # Process Total
        total_results = self._analyze_total(totals) if totals else {}
        
        # Process Moneyline
        ml_results = self._analyze_moneyline(ml_away_series, ml_home_series) if ml_away_series or ml_home_series else {}
        
        return {
            'spread': spread_results,
            'total': total_results,
            'moneyline': ml_results,
            'data': parsed_data
        }
    
    def _analyze_spread(self, spreads: List[float]) -> Dict[str, Any]:
        """Analyze spread data"""
        # Calculate all indicators
        sma_spread = self._calculate_sma(spreads, 10)
        ema_spread = self._calculate_ema(spreads, 5)
        rsi_spread = self._calculate_rsi(spreads, 7)
        macd_spread = self._calculate_macd(spreads)
        bb_spread = self._calculate_bollinger_bands(spreads, 20, 2)
        atr_spread = self._calculate_atr(spreads, 14)
        z_spread = self._calculate_zscore(spreads, 10)
        fib_spread = self._calculate_fibonacci_levels(spreads, 13)
        greeks_spread = self._calculate_greeks(spreads, 13)
        steam_spread = self._detect_steam_moves(spreads, 2)
        roc_spread = self._calculate_roc(spreads, 2)
        ama_spread = self._calculate_adaptive_ma(spreads, 2, 8, 6)
        
        return {
            'sma': sma_spread,
            'ema': ema_spread,
            'rsi': rsi_spread,
            'macd': macd_spread,
            'bollinger_bands': bb_spread,
            'atr': atr_spread,
            'zscore': z_spread,
            'fibonacci': fib_spread,
            'greeks': greeks_spread,
            'steam_moves': steam_spread,
            'roc': roc_spread,
            'adaptive_ma': ama_spread
        }
    
    def _analyze_total(self, totals: List[float]) -> Dict[str, Any]:
        """Analyze total data"""
        # Calculate all indicators
        sma_total = self._calculate_sma(totals, 10)
        ema_total = self._calculate_ema(totals, 5)
        rsi_total = self._calculate_rsi(totals, 7)
        macd_total = self._calculate_macd(totals)
        bb_total = self._calculate_bollinger_bands(totals, 20, 2)
        atr_total = self._calculate_atr(totals, 14)
        z_total = self._calculate_zscore(totals, 10)
        fib_total = self._calculate_fibonacci_levels(totals, 13)
        greeks_total = self._calculate_greeks(totals, 13)
        steam_total = self._detect_steam_moves(totals, 2)
        roc_total = self._calculate_roc(totals, 2)
        ama_total = self._calculate_adaptive_ma(totals, 2, 8, 6)
        
        return {
            'sma': sma_total,
            'ema': ema_total,
            'rsi': rsi_total,
            'macd': macd_total,
            'bollinger_bands': bb_total,
            'atr': atr_total,
            'zscore': z_total,
            'fibonacci': fib_total,
            'greeks': greeks_total,
            'steam_moves': steam_total,
            'roc': roc_total,
            'adaptive_ma': ama_total
        }
    
    def _analyze_moneyline(self, ml_away: List, ml_home: List) -> Dict[str, Any]:
        """Analyze moneyline data"""
        # Calculate all indicators
        sma_ml_away = self._calculate_sma(ml_away, 10) if ml_away else []
        ema_ml_away = self._calculate_ema(ml_away, 5) if ml_away else []
        rsi_ml_away = self._calculate_rsi(ml_away, 7) if ml_away else []
        macd_ml_away = self._calculate_macd(ml_away) if ml_away else {}
        bb_ml_away = self._calculate_bollinger_bands(ml_away, 20, 2) if ml_away else []
        atr_ml_away = self._calculate_atr(ml_away, 14) if ml_away else None
        z_ml_away = self._calculate_zscore(ml_away, 10) if ml_away else []
        fib_ml_away = self._calculate_fibonacci_levels(ml_away, 13) if ml_away else None
        greeks_ml_away = self._calculate_greeks(ml_away, 13) if ml_away else None
        steam_ml_away = self._detect_steam_moves(ml_away, 20) if ml_away else []
        roc_ml_away = self._calculate_roc(ml_away, 2) if ml_away else []
        ama_ml_away = self._calculate_adaptive_ma(ml_away, 2, 8, 6) if ml_away else []
        
        return {
            'sma_away': sma_ml_away,
            'ema_away': ema_ml_away,
            'rsi_away': rsi_ml_away,
            'macd_away': macd_ml_away,
            'bollinger_bands_away': bb_ml_away,
            'atr_away': atr_ml_away,
            'zscore_away': z_ml_away,
            'fibonacci_away': fib_ml_away,
            'greeks_away': greeks_ml_away,
            'steam_moves_away': steam_ml_away,
            'roc_away': roc_ml_away,
            'adaptive_ma_away': ama_ml_away
        }
    
    def _calculate_sma(self, values: List[float], period: int) -> List[Optional[float]]:
        """Simple Moving Average"""
        if len(values) < period:
            return [None] * len(values)
        
        result = [None] * (period - 1)
        for i in range(period - 1, len(values)):
            result.append(sum(values[i - period + 1:i + 1]) / period)
        return result
    
    def _calculate_ema(self, values: List[float], period: int) -> List[Optional[float]]:
        """Exponential Moving Average"""
        if len(values) < period:
            return [None] * len(values)
        
        result = [None] * (period - 1)
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        result.append(ema)
        
        for i in range(period, len(values)):
            ema = values[i] * k + ema * (1 - k)
            result.append(ema)
        return result
    
    def _calculate_adaptive_ma(self, values: List[float], fast: int = 2, slow: int = 10, efficiency_lookback: int = 8) -> List[Optional[float]]:
        """Adaptive Moving Average"""
        if len(values) < slow + efficiency_lookback:
            return [None] * len(values)
        
        result = [None] * (slow + efficiency_lookback - 1)
        for i in range(slow + efficiency_lookback - 1, len(values)):
            change = abs(values[i] - values[i - efficiency_lookback])
            volatility = sum(abs(values[j] - values[j-1]) for j in range(i - efficiency_lookback + 1, i + 1))
            
            if volatility == 0:
                er = 0
            else:
                er = change / volatility
            
            fast_sc = 2 / (fast + 1)
            slow_sc = 2 / (slow + 1)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            if result[i-1] is None:
                result[i-1] = values[i - efficiency_lookback]
            
            result.append(result[i-1] + sc * (values[i] - result[i-1]))
        return result
    
    def _calculate_fibonacci_levels(self, values: List[float], window: int = 13) -> Optional[Dict[str, Any]]:
        """Calculate Fibonacci retracement and extension levels"""
        if len(values) < window:
            return None
        
        slice_values = values[-window:]
        high = max(slice_values)
        low = min(slice_values)
        
        retracements = [0.236, 0.382, 0.5, 0.618, 0.786]
        extensions = [1.236, 1.382, 1.5, 1.618, 2.0]
        
        retracement_levels = [high - (high - low) * r for r in retracements]
        extension_levels = [high + (high - low) * (e - 1) for e in extensions]
        
        return {
            'high': high,
            'low': low,
            'retracements': retracement_levels,
            'extensions': extension_levels
        }
    
    def _calculate_roc(self, values: List[float], period: int = 2) -> List[Optional[float]]:
        """Rate of Change"""
        result = [None] * period
        for i in range(period, len(values)):
            if values[i - period] != 0:
                result.append(100 * (values[i] - values[i - period]) / abs(values[i - period]))
            else:
                result.append(None)
        return result
    
    def _calculate_rsi(self, values: List[float], period: int = 5) -> List[Optional[float]]:
        """Relative Strength Index"""
        if len(values) < period + 1:
            return [None] * len(values)
        
        result = [None] * (period + 1)
        gains = 0
        losses = 0
        
        # Calculate initial gains and losses
        for i in range(1, period + 1):
            diff = values[i] - values[i-1]
            if diff > 0:
                gains += diff
            else:
                losses += abs(diff)
        
        avg_g = gains / period
        avg_l = losses / period
        
        if avg_l != 0:
            rs = avg_g / avg_l
            result[period] = 100 - (100 / (1 + rs))
        else:
            result[period] = 100
        
        # Calculate subsequent RSI values
        for i in range(period + 2, len(values)):
            diff = values[i] - values[i-1]
            if diff > 0:
                avg_g = (avg_g * (period - 1) + diff) / period
                avg_l = (avg_l * (period - 1)) / period
            else:
                avg_g = (avg_g * (period - 1)) / period
                avg_l = (avg_l * (period - 1) + abs(diff)) / period
            
            if avg_l != 0:
                rs = avg_g / avg_l
                result[i] = 100 - (100 / (1 + rs))
            else:
                result[i] = 100
        
        return result
    
    def _calculate_macd(self, values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[Optional[float]]]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = self._calculate_ema(values, fast)
        ema_slow = self._calculate_ema(values, slow)
        
        macd = []
        for i in range(len(values)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd.append(ema_fast[i] - ema_slow[i])
            else:
                macd.append(None)
        
        signal_line = self._calculate_ema([x for x in macd if x is not None], signal)
        signal_full = [None] * len(macd)
        
        # Align signal with MACD
        offset = next(i for i, x in enumerate(macd) if x is not None)
        for i in range(len(signal_line)):
            signal_full[offset + i] = signal_line[i]
        
        return {'macd': macd, 'signal': signal_full}
    
    def _calculate_bollinger_bands(self, values: List[float], period: int = 20, mult: int = 2) -> List[Dict[str, float]]:
        """Bollinger Bands"""
        if len(values) < period:
            return [{}] * len(values)
        
        result = [{}] * (period - 1)
        for i in range(period - 1, len(values)):
            slice_values = values[i - period + 1:i + 1]
            sma = sum(slice_values) / period
            std = math.sqrt(sum((x - sma) ** 2 for x in slice_values) / period)
            
            result.append({
                'upper': sma + mult * std,
                'lower': sma - mult * std
            })
        return result
    
    def _calculate_atr(self, values: List[float], period: int = 14) -> Optional[float]:
        """Average True Range"""
        if len(values) < period + 1:
            return None
        
        tr = []
        for i in range(1, len(values)):
            tr.append(abs(values[i] - values[i-1]))
        
        # Calculate EMA of TR
        k = 2 / (period + 1)
        ema = sum(tr[:period]) / period
        for i in range(period, len(tr)):
            ema = tr[i] * k + ema * (1 - k)
        
        return ema
    
    def _calculate_zscore(self, values: List[float], period: int = 10) -> List[Optional[float]]:
        """Z-score"""
        if len(values) < period:
            return [None] * len(values)
        
        result = [None] * (period - 1)
        for i in range(period - 1, len(values)):
            slice_values = values[i - period + 1:i + 1]
            mean = sum(slice_values) / period
            std = math.sqrt(sum((x - mean) ** 2 for x in slice_values) / period)
            
            if std != 0:
                result.append((values[i] - mean) / std)
            else:
                result.append(0.0)
        return result
    
    def _calculate_greeks(self, values: List[float], window: int = 13) -> Optional[Dict[str, float]]:
        """Greek analysis (delta, gamma, vega, theta)"""
        if len(values) < window:
            return None
        
        delta = values[-1] - values[-2]
        prev_delta = values[-2] - values[-3]
        gamma = delta - prev_delta
        
        slice_values = values[-window:]
        mean = sum(slice_values) / window
        vega = math.sqrt(sum((x - mean) ** 2 for x in slice_values) / window)
        theta = (values[-1] - values[-window]) / window
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }
    
    def _detect_steam_moves(self, values: List[float], threshold: float = 2) -> List[Dict[str, Any]]:
        """Detect significant price movements"""
        moves = []
        for i in range(1, len(values)):
            if abs(values[i] - values[i-1]) >= threshold:
                moves.append({
                    'index': i,
                    'from': values[i-1],
                    'to': values[i],
                    'change': values[i] - values[i-1]
                })
        return moves
