import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats, optimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from scipy.stats import norm, zscore
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

warnings.filterwarnings('ignore')

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0-100
    probability: float  # 0-1
    confidence: float  # 0-1
    price_level: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    indicators: Dict[str, float]
    narrative: str

class AdvancedTechnicalAnalysis:
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.adaptive_ma_history = []
        self.setup_ml_models()
        
    def setup_ml_models(self):
        """Initialize machine learning models for pattern recognition"""
        self.spread_model = GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.total_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1, 
            max_depth=6,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_trained = False
        
    def calculate_adaptive_ma(self, data: pd.Series, efficiency_period: int = 10, 
                            fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
        """Advanced Adaptive Moving Average with dynamic smoothing"""
        if len(data) < efficiency_period:
            return pd.Series([np.nan] * len(data), index=data.index)
            
        # Calculate efficiency ratio
        change = abs(data - data.shift(efficiency_period))
        volatility = data.rolling(window=efficiency_period).std() * np.sqrt(efficiency_period)
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate smoothing constant
        fast_sc = 2 / (fast_ema + 1)
        slow_sc = 2 / (slow_ema + 1)
        smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate adaptive MA
        adaptive_ma = data.copy()
        for i in range(1, len(data)):
            if not pd.isna(smoothing_constant.iloc[i]) and not pd.isna(adaptive_ma.iloc[i-1]):
                adaptive_ma.iloc[i] = (smoothing_constant.iloc[i] * data.iloc[i] + 
                                     (1 - smoothing_constant.iloc[i]) * adaptive_ma.iloc[i-1])
        
        return adaptive_ma

    def calculate_kalman_filter(self, data: pd.Series, process_variance: float = 1e-5, 
                              measurement_variance: float = 0.1) -> pd.Series:
        """Kalman Filter for optimal estimation"""
        estimates = []
        estimate = data.iloc[0] if len(data) > 0 else 0
        error_estimate = 1.0
        
        for value in data:
            # Prediction update
            error_estimate += process_variance
            
            # Measurement update
            kalman_gain = error_estimate / (error_estimate + measurement_variance)
            estimate = estimate + kalman_gain * (value - estimate)
            error_estimate = (1 - kalman_gain) * error_estimate
            estimates.append(estimate)
            
        return pd.Series(estimates, index=data.index)

    def calculate_fibonacci_retracement(self, high: float, low: float) -> Dict[str, float]:
        """Complete Fibonacci retracement levels"""
        diff = high - low
        return {
            '0.0': high,
            '0.236': high - 0.236 * diff,
            '0.382': high - 0.382 * diff, 
            '0.5': high - 0.5 * diff,
            '0.618': high - 0.618 * diff,
            '0.786': high - 0.786 * diff,
            '1.0': low
        }

    def calculate_fibonacci_extension(self, high: float, low: float) -> Dict[str, float]:
        """Complete Fibonacci extension levels"""
        diff = high - low
        return {
            '1.272': high + 0.272 * diff,
            '1.414': high + 0.414 * diff,
            '1.618': high + 0.618 * diff,
            '2.0': high + diff,
            '2.272': high + 1.272 * diff,
            '2.618': high + 1.618 * diff
        }

    def calculate_greek_analysis(self, data: pd.Series) -> Dict[str, float]:
        """Comprehensive Greek analysis for options-like sensitivity"""
        if len(data) < 10:
            return {}
            
        # Delta - first derivative (slope)
        prices = data.values
        delta = (prices[-1] - prices[-2]) if len(prices) > 1 else 0
        
        # Gamma - second derivative (acceleration)
        gamma = (prices[-1] - 2*prices[-2] + prices[-3]) if len(prices) > 2 else 0
        
        # Theta - time decay
        theta = (prices[-1] - prices[-5]) / 4 if len(prices) > 5 else 0
        
        # Vega - volatility sensitivity
        vega = np.std(prices[-10:]) if len(prices) >= 10 else 0
        
        # Rho - correlation with market (simplified)
        rho = np.corrcoef(prices[-10:], range(10))[0,1] if len(prices) >= 10 else 0
        
        return {
            'delta': delta,
            'gamma': gamma, 
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def calculate_advanced_rsi(self, data: pd.Series, period: int = 14) -> Dict[str, Any]:
        """Advanced RSI with divergence detection"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # RSI divergence detection
        price_slope = np.polyfit(range(5), data.tail(5).values, 1)[0] if len(data) >= 5 else 0
        rsi_slope = np.polyfit(range(5), rsi.tail(5).values, 1)[0] if len(rsi) >= 5 else 0
        
        divergence = "bullish" if price_slope < 0 and rsi_slope > 0 else \
                    "bearish" if price_slope > 0 and rsi_slope < 0 else "none"
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'divergence': divergence,
            'momentum': rsi_slope,
            'overbought': rsi.iloc[-1] > 70 if not rsi.empty else False,
            'oversold': rsi.iloc[-1] < 30 if not rsi.empty else False
        }

    def calculate_macd_advanced(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
        """Advanced MACD with histogram analysis"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # MACD analysis
        histogram_slope = np.polyfit(range(3), histogram.tail(3).values, 1)[0] if len(histogram) >= 3 else 0
        macd_cross = "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2] else \
                    "bearish" if macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2] else "none"
        
        return {
            'macd': macd_line.iloc[-1] if not macd_line.empty else 0,
            'signal': signal_line.iloc[-1] if not signal_line.empty else 0,
            'histogram': histogram.iloc[-1] if not histogram.empty else 0,
            'histogram_slope': histogram_slope,
            'cross': macd_cross,
            'momentum': 'rising' if histogram_slope > 0 else 'falling'
        }

    def calculate_bollinger_bands_advanced(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, Any]:
        """Advanced Bollinger Bands analysis"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Band analysis
        bandwidth = (upper_band - lower_band) / sma * 100
        bb_position = (data.iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) if not upper_band.empty else 0.5
        
        squeeze = bandwidth.iloc[-1] < bandwidth.rolling(20).mean().iloc[-1] if len(bandwidth) >= 20 else False
        
        return {
            'upper': upper_band.iloc[-1] if not upper_band.empty else data.iloc[-1],
            'middle': sma.iloc[-1] if not sma.empty else data.iloc[-1],
            'lower': lower_band.iloc[-1] if not lower_band.empty else data.iloc[-1],
            'bandwidth': bandwidth.iloc[-1] if not bandwidth.empty else 0,
            'position': bb_position,
            'squeeze': squeeze,
            'breakout_potential': 'high' if squeeze else 'normal'
        }

    def calculate_atr_advanced(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, float]:
        """Advanced Average True Range analysis"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return {
            'atr': atr.iloc[-1] if not atr.empty else 0,
            'atr_percent': (atr.iloc[-1] / close.iloc[-1]) * 100 if not atr.empty and close.iloc[-1] != 0 else 0,
            'volatility': 'high' if atr.iloc[-1] > atr.rolling(20).mean().iloc[-1] else 'low' if len(atr) >= 20 else 'normal'
        }

    def train_ml_models(self, spread_data: pd.DataFrame, total_data: pd.DataFrame, target_data: pd.Series):
        """Train machine learning models on historical data"""
        try:
            # Feature engineering
            spread_features = self.create_ml_features(spread_data, 'spread')
            total_features = self.create_ml_features(total_data, 'total')
            
            # Combine features
            features = pd.concat([spread_features, total_features], axis=1)
            features = features.dropna()
            
            if len(features) < 50:  # Minimum samples required
                print("Insufficient data for training")
                return
                
            # Prepare targets
            targets = target_data.loc[features.index]
            
            # Remove any remaining NaN values
            valid_idx = targets.notna() & features.notna().all(axis=1)
            features = features[valid_idx]
            targets = targets[valid_idx]
            
            if len(features) < 20:
                print("Not enough valid samples after cleaning")
                return
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.spread_model.fit(features_scaled, targets)
            self.model_trained = True
            
            # Store feature importance
            self.feature_importance = dict(zip(features.columns, 
                                            self.spread_model.feature_importances_))
            
            print(f"Model trained successfully on {len(features)} samples")
            print(f"Feature importance: {self.feature_importance}")
            
        except Exception as e:
            print(f"Error training ML model: {e}")

    def create_ml_features(self, data: pd.Series, prefix: str) -> pd.DataFrame:
        """Create comprehensive features for ML model"""
        features = {}
        
        # Price-based features
        features[f'{prefix}_returns_1'] = data.pct_change(1)
        features[f'{prefix}_returns_5'] = data.pct_change(5)
        features[f'{prefix}_volatility_10'] = data.rolling(10).std()
        
        # Technical indicators as features
        features[f'{prefix}_rsi_14'] = self.calculate_advanced_rsi(data)['rsi']
        macd = self.calculate_macd_advanced(data)
        features[f'{prefix}_macd'] = macd['macd']
        features[f'{prefix}_macd_signal'] = macd['signal']
        
        bb = self.calculate_bollinger_bands_advanced(data)
        features[f'{prefix}_bb_position'] = bb['position']
        features[f'{prefix}_bb_bandwidth'] = bb['bandwidth']
        
        # Statistical features
        features[f'{prefix}_zscore_20'] = zscore(data.tail(20))[-1] if len(data) >= 20 else 0
        features[f'{prefix}_skew_20'] = data.tail(20).skew() if len(data) >= 20 else 0
        features[f'{prefix}_kurtosis_20'] = data.tail(20).kurtosis() if len(data) >= 20 else 0
        
        return pd.DataFrame(features)

    def generate_trading_signals(self, current_data: Dict[str, float], 
                               historical_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate comprehensive trading signals"""
        signals = []
        
        # Analyze spreads
        spread_signals = self.analyze_spread_movement(historical_data['spread'])
        signals.extend(spread_signals)
        
        # Analyze totals
        total_signals = self.analyze_total_movement(historical_data['total'])
        signals.extend(total_signals)
        
        return signals

    def analyze_spread_movement(self, spread_data: pd.Series) -> List[TradingSignal]:
        """Advanced spread movement analysis"""
        if len(spread_data) < 20:
            return []
            
        current_spread = spread_data.iloc[-1]
        
        # Calculate all technical indicators
        adaptive_ma = self.calculate_adaptive_ma(spread_data)
        kalman_filter = self.calculate_kalman_filter(spread_data)
        rsi_analysis = self.calculate_advanced_rsi(spread_data)
        macd_analysis = self.calculate_macd_advanced(spread_data)
        bb_analysis = self.calculate_bollinger_bands_advanced(spread_data)
        greek_analysis = self.calculate_greek_analysis(spread_data)
        
        # Fibonacci levels
        high = spread_data.tail(20).max()
        low = spread_data.tail(20).min()
        fib_retracement = self.calculate_fibonacci_retracement(high, low)
        fib_extension = self.calculate_fibonacci_extension(high, low)
        
        # Generate signal strength and probability
        signal_strength, probability = self.calculate_signal_strength(
            spread_data, rsi_analysis, macd_analysis, bb_analysis
        )
        
        # Create narrative
        narrative = self.generate_spread_narrative(
            current_spread, rsi_analysis, macd_analysis, bb_analysis, fib_retracement
        )
        
        signal = TradingSignal(
            symbol="SPREAD",
            signal_type="BUY" if signal_strength > 0 else "SELL",
            strength=abs(signal_strength),
            probability=probability,
            confidence=min(probability * 1.2, 0.95),
            price_level=current_spread,
            stop_loss=current_spread * 0.98,
            take_profit=current_spread * 1.02,
            timestamp=datetime.now(),
            indicators={
                'adaptive_ma': adaptive_ma.iloc[-1] if not adaptive_ma.empty else current_spread,
                'kalman_filter': kalman_filter.iloc[-1] if not kalman_filter.empty else current_spread,
                'rsi': rsi_analysis['rsi'],
                'macd': macd_analysis['macd'],
                'bb_position': bb_analysis['position'],
                'delta': greek_analysis.get('delta', 0)
            },
            narrative=narrative
        )
        
        return [signal]

    def analyze_total_movement(self, total_data: pd.Series) -> List[TradingSignal]:
        """Advanced total movement analysis"""
        if len(total_data) < 20:
            return []
            
        current_total = total_data.iloc[-1]
        
        # Technical analysis
        adaptive_ma = self.calculate_adaptive_ma(total_data)
        rsi_analysis = self.calculate_advanced_rsi(total_data)
        macd_analysis = self.calculate_macd_advanced(total_data)
        bb_analysis = self.calculate_bollinger_bands_advanced(total_data)
        
        # Generate signal
        signal_strength, probability = self.calculate_signal_strength(
            total_data, rsi_analysis, macd_analysis, bb_analysis
        )
        
        narrative = self.generate_total_narrative(
            current_total, rsi_analysis, macd_analysis, bb_analysis
        )
        
        signal = TradingSignal(
            symbol="TOTAL",
            signal_type="OVER" if signal_strength > 0 else "UNDER",
            strength=abs(signal_strength),
            probability=probability,
            confidence=min(probability * 1.1, 0.90),
            price_level=current_total,
            stop_loss=current_total * 0.99,
            take_profit=current_total * 1.01,
            timestamp=datetime.now(),
            indicators={
                'adaptive_ma': adaptive_ma.iloc[-1] if not adaptive_ma.empty else current_total,
                'rsi': rsi_analysis['rsi'],
                'macd': macd_analysis['macd'],
                'bb_position': bb_analysis['position']
            },
            narrative=narrative
        )
        
        return [signal]

    def calculate_signal_strength(self, data: pd.Series, rsi_analysis: Dict, 
                                macd_analysis: Dict, bb_analysis: Dict) -> Tuple[float, float]:
        """Calculate signal strength and probability"""
        strength = 0
        probability_factors = []
        
        # RSI factor
        rsi = rsi_analysis['rsi']
        if rsi < 30:
            strength += 25
            probability_factors.append(0.8)
        elif rsi > 70:
            strength -= 25
            probability_factors.append(0.8)
        else:
            probability_factors.append(0.5)
            
        # MACD factor
        if macd_analysis['cross'] == 'bullish':
            strength += 20
            probability_factors.append(0.7)
        elif macd_analysis['cross'] == 'bearish':
            strength -= 20
            probability_factors.append(0.7)
        else:
            probability_factors.append(0.5)
            
        # Bollinger Bands factor
        bb_pos = bb_analysis['position']
        if bb_pos < 0.2:
            strength += 15
            probability_factors.append(0.6)
        elif bb_pos > 0.8:
            strength -= 15
            probability_factors.append(0.6)
        else:
            probability_factors.append(0.5)
            
        # Trend factor
        trend = np.polyfit(range(10), data.tail(10).values, 1)[0] if len(data) >= 10 else 0
        if trend > 0:
            strength += 10
            probability_factors.append(0.6)
        else:
            strength -= 10
            probability_factors.append(0.6)
            
        # Normalize strength to -100 to 100 range
        strength = max(min(strength, 100), -100)
        
        # Calculate probability as average of factors
        probability = np.mean(probability_factors) if probability_factors else 0.5
        
        return strength, probability

    def generate_spread_narrative(self, current_spread: float, rsi_analysis: Dict,
                                macd_analysis: Dict, bb_analysis: Dict, fib_levels: Dict) -> str:
        """Generate professional betting narrative for spreads"""
        
        narratives = []
        
        # RSI narrative
        if rsi_analysis['rsi'] < 30:
            narratives.append("RSI indicates severely oversold conditions")
        elif rsi_analysis['rsi'] > 70:
            narratives.append("RSI shows overbought territory")
        else:
            narratives.append("RSI in neutral range")
            
        # MACD narrative
        if macd_analysis['cross'] == 'bullish':
            narratives.append("MACD bullish crossover detected")
        elif macd_analysis['cross'] == 'bearish':
            narratives.append("MACD bearish crossover confirmed")
            
        # Bollinger Bands narrative
        if bb_analysis['position'] < 0.2:
            narratives.append("Trading near lower Bollinger Band support")
        elif bb_analysis['position'] > 0.8:
            narratives.append("Approaching upper Bollinger Band resistance")
            
        # Fibonacci narrative
        closest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_spread))
        narratives.append(f"Near Fibonacci {closest_fib[0]} level at {closest_fib[1]:.2f}")
        
        # Combine narratives
        base_narrative = f"Spread at {current_spread:.2f} showing "
        if len(narratives) > 0:
            base_narrative += ", ".join(narratives)
        else:
            base_narrative += "neutral technical characteristics"
            
        return base_narrative + ". Monitor for confirmation signals."

    def generate_total_narrative(self, current_total: float, rsi_analysis: Dict,
                               macd_analysis: Dict, bb_analysis: Dict) -> str:
        """Generate professional betting narrative for totals"""
        
        narratives = []
        
        # Market pressure analysis
        if rsi_analysis['rsi'] > 60:
            narratives.append("bullish momentum building")
        elif rsi_analysis['rsi'] < 40:
            narratives.append("bearish pressure increasing")
            
        if macd_analysis['histogram_slope'] > 0:
            narratives.append("upward momentum accelerating")
        elif macd_analysis['histogram_slope'] < 0:
            narratives.append("momentum slowing")
            
        if bb_analysis['squeeze']:
            narratives.append("Bollinger Band squeeze suggesting impending breakout")
            
        base_narrative = f"Total at {current_total:.2f} showing "
        if len(narratives) > 0:
            base_narrative += ", ".join(narratives)
        else:
            base_narrative += "consolidation patterns"
            
        return base_narrative + ". Await volume confirmation for direction."

    def generate_comprehensive_report(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Generate complete analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'signals': [],
            'summary': {
                'total_signals': len(signals),
                'strong_signals': len([s for s in signals if s.strength > 70]),
                'average_confidence': np.mean([s.confidence for s in signals]) if signals else 0,
                'market_outlook': self.assess_market_outlook(signals)
            },
            'adaptive_ma_analysis': self.analyze_adaptive_ma_trend(),
            'risk_assessment': self.calculate_portfolio_risk(signals)
        }
        
        for signal in signals:
            report['signals'].append({
                'symbol': signal.symbol,
                'signal': signal.signal_type,
                'strength': signal.strength,
                'probability': signal.probability,
                'narrative': signal.narrative,
                'indicators': signal.indicators
            })
            
        return report

    def assess_market_outlook(self, signals: List[TradingSignal]) -> str:
        """Assess overall market outlook based on signals"""
        if not signals:
            return "NEUTRAL"
            
        avg_strength = np.mean([s.strength for s in signals])
        bullish_count = len([s for s in signals if s.strength > 0])
        bearish_count = len([s for s in signals if s.strength < 0])
        
        if bullish_count > bearish_count and avg_strength > 30:
            return "BULLISH"
        elif bearish_count > bullish_count and avg_strength < -30:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def analyze_adaptive_ma_trend(self) -> Dict[str, Any]:
        """Analyze Adaptive MA trend for visualization"""
        if len(self.adaptive_ma_history) < 10:
            return {'trend': 'INSUFFICIENT_DATA', 'slope': 0}
            
        recent_ma = self.adaptive_ma_history[-10:]
        slope = np.polyfit(range(10), recent_ma, 1)[0]
        
        return {
            'trend': 'UP' if slope > 0.01 else 'DOWN' if slope < -0.01 else 'FLAT',
            'slope': slope,
            'current_value': recent_ma[-1] if recent_ma else 0,
            'history': self.adaptive_ma_history[-20:]  # Last 20 values for display
        }

    def calculate_portfolio_risk(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        if not signals:
            return {'total_risk': 0, 'max_drawdown': 0, 'sharpe_ratio': 0}
            
        confidences = [s.confidence for s in signals]
        strengths = [abs(s.strength) for s in signals]
        
        return {
            'total_risk': np.std(confidences) if len(confidences) > 1 else 0,
            'max_drawdown': min(strengths) if strengths else 0,
            'sharpe_ratio': np.mean(strengths) / np.std(strengths) if len(strengths) > 1 and np.std(strengths) != 0 else 0,
            'win_probability': np.mean([s.probability for s in signals]) if signals else 0
        }

# Example usage with real data simulation
def demonstrate_ta_engine():
    """Demonstrate the complete TA engine with sample data"""
    
    # Generate sample sports betting data
    dates = pd.date_range(start='2024-01-01', end='2024-01-20', freq='H')
    np.random.seed(42)
    
    # Simulate spread data (signed values)
    spread_data = pd.Series(
        np.cumsum(np.random.normal(0, 0.5, len(dates))) + 3.0,
        index=dates
    )
    
    # Simulate total data
    total_data = pd.Series(
        np.cumsum(np.random.normal(0, 0.3, len(dates))) + 45.0,
        index=dates
    )
    
    # Create TA engine instance
    ta_engine = AdvancedTechnicalAnalysis(lookback_period=100)
    
    # Generate signals
    historical_data = pd.DataFrame({
        'spread': spread_data,
        'total': total_data
    })
    
    signals = ta_engine.generate_trading_signals(
        current_data={'spread': spread_data.iloc[-1], 'total': total_data.iloc[-1]},
        historical_data=historical_data
    )
    
    # Generate comprehensive report
    report = ta_engine.generate_comprehensive_report(signals)
    
    print("=== ADVANCED TECHNICAL ANALYSIS ENGINE ===")
    print(f"Generated {len(signals)} trading signals")
    print(f"Market Outlook: {report['summary']['market_outlook']}")
    print(f"Average Confidence: {report['summary']['average_confidence']:.2f}")
    print("\n--- Detailed Signals ---")
    
    for signal in report['signals']:
        print(f"\n{signal['symbol']}: {signal['signal']} (Strength: {signal['strength']:.1f})")
        print(f"Probability: {signal['probability']:.2f}")
        print(f"Analysis: {signal['narrative']}")
        print(f"Key Indicators: RSI: {signal['indicators'].get('rsi', 'N/A'):.1f}, "
              f"MACD: {signal['indicators'].get('macd', 'N/A'):.3f}")
    
    print(f"\n--- Adaptive MA Analysis ---")
    ma_analysis = report['adaptive_ma_analysis']
    print(f"Trend: {ma_analysis['trend']}, Slope: {ma_analysis['slope']:.4f}")
    
    print(f"\n--- Risk Assessment ---")
    risk = report['risk_assessment']
    print(f"Total Risk: {risk['total_risk']:.3f}, Win Probability: {risk['win_probability']:.2f}")
    
    return ta_engine, report

if __name__ == "__main__":
    # Run the complete demonstration
    engine, report = demonstrate_ta_engine()
    
    # Save report to JSON for inspection
    with open('ta_engine_report.json', 'w') as f:
        json_report = json.loads(json.dumps(report, default=str))
        json.dump(json_report, f, indent=2)
    
    print("\nReport saved to 'ta_engine_report.json'")
