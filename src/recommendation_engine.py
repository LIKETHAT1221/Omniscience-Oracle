from typing import Dict, List
import numpy as np

class RecommendationEngine:
    def __init__(self):
        self.confidence_thresholds = {
            'strong': 0.7,
            'moderate': 0.6,
            'weak': 0.55
        }
        
    def generate_recommendation(self, game_data: Dict, ta_indicators: Dict) -> Dict:
        confidence = 0.5
        narrative = []
        edge_strength = 0.0
        
        rlm_strength = self._calculate_rlm_strength(game_data)
        if rlm_strength is not None:
            if abs(rlm_strength) > 10:
                away_bets = game_data.get('away_bets_pct', 0)
                home_bets = game_data.get('home_bets_pct', 0)
                away_money = game_data.get('away_money_pct', 0)
                home_money = game_data.get('home_money_pct', 0)
                
                if away_bets > home_bets:
                    narrative.append(
                        f"Strong RLM detected: {away_bets:.0f}% bets on away, "
                        f"but {home_money:.0f}% money on home. Sharp money fading public."
                    )
                else:
                    narrative.append(
                        f"Strong RLM detected: {home_bets:.0f}% bets on home, "
                        f"but {away_money:.0f}% money on away. Sharp money fading public."
                    )
                
                if rlm_strength > 0:
                    confidence += 0.25
                else:
                    confidence += 0.15
            else:
                narrative.append("No significant RLM detected.")
        else:
            narrative.append("No splits data available for RLM analysis.")
        
        ta_analysis = self._analyze_ta_indicators(ta_indicators)
        confidence += ta_analysis['confidence_impact']
        narrative.extend(ta_analysis['narrative'])
        
        recommendation, strength = self._determine_recommendation(confidence, ta_analysis)
        
        expected_value = self._calculate_expected_value(confidence, ta_analysis)
        kelly_stake = self._calculate_kelly_stake(expected_value)
        
        return {
            'market_type': game_data.get('game_type', 'unknown'),
            'recommendation': f"{strength}{recommendation}",
            'confidence': confidence,
            'narrative': " ".join(narrative),
            'expected_value': expected_value,
            'kelly_stake': kelly_stake,
            'timestamp': game_data.get('parsed_at', '')
        }
        
    def _calculate_rlm_strength(self, game_data: Dict) -> float:
        away_bets = game_data.get('away_bets_pct')
        home_bets = game_data.get('home_bets_pct')
        away_money = game_data.get('away_money_pct')
        home_money = game_data.get('home_money_pct')
        
        if None in [away_bets, home_bets, away_money, home_money]:
            return None
            
        sentiment_imb = away_bets - home_bets
        money_imb = away_money - home_money
        return money_imb - sentiment_imb
        
    def _analyze_ta_indicators(self, ta_indicators: Dict) -> Dict:
        confidence_impact = 0.0
        narrative = []
        
        for market, indicators in ta_indicators.items():
            if 'momentum' in indicators and indicators['momentum']['MOM_V'] is not None:
                mom_v = indicators['momentum']['MOM_V']
                mom_a = indicators['momentum']['MOM_A']
                
                narrative.append(f"{market}: MOM_V={mom_v:.4f}, MOM_A={mom_a:.4f if mom_a is not None else 'N/A'}")
                
                if mom_v > 0 and mom_a is not None and mom_a > 0:
                    confidence_impact += 0.15
                    narrative.append(f"Strong bullish momentum detected in {market}.")
                elif mom_v < 0 and mom_a is not None and mom_a < 0:
                    confidence_impact -= 0.15
                    narrative.append(f"Strong bearish momentum detected in {market}.")
                    
            if 'rsi' in indicators and indicators['rsi'] is not None:
                rsi = indicators['rsi']
                if rsi > 70:
                    confidence_impact -= 0.1
                    narrative.append(f"{market} is overbought (RSI: {rsi:.1f}).")
                elif rsi < 30:
                    confidence_impact += 0.1
                    narrative.append(f"{market} is oversold (RSI: {rsi:.1f}).")
                    
            if 'adaptive_ma' in indicators and indicators['adaptive_ma'] is not None:
                adaptive_ma = indicators['adaptive_ma']
                current_val = indicators.get('current_value')
                if current_val is not None:
                    edge = adaptive_ma - current_val
                    if abs(edge) > 0.02:
                        confidence_impact += edge * 5
                        direction = "above" if edge > 0 else "below"
                        narrative.append(
                            f"Adaptive MA suggests true value is {abs(edge):.2%} {direction} current {market} value."
                        )
        
        return {
            'confidence_impact': confidence_impact,
            'narrative': narrative
        }
        
    def _determine_recommendation(self, confidence: float, ta_analysis: Dict) -> tuple:
        if confidence >= self.confidence_thresholds['strong']:
            return "BACK", "STRONG "
        elif confidence >= self.confidence_thresholds['moderate']:
            return "BACK", ""
        elif confidence >= self.confidence_thresholds['weak']:
            return "FADE", ""
        else:
            return "HOLD", ""
            
    def _calculate_expected_value(self, confidence: float, ta_analysis: Dict) -> float:
        base_ev = (confidence - 0.5) * 0.15
        return max(-0.1, min(0.2, base_ev))
        
    def _calculate_kelly_stake(self, expected_value: float) -> float:
        if expected_value <= 0:
            return 0.0
            
        odds = 1.91
        kelly = expected_value / (odds - 1)
        
        return max(0.0, min(0.1, kelly / 4))
