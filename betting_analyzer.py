import math
from typing import List, Dict, Tuple, Optional, Any

class BettingAnalyzer:
    """Betting analysis with EV and Kelly Criterion"""
    
    def __init__(self):
        pass
    
    def analyze_all(self, parsed_data: List[Dict]) -> Dict[str, Any]:
        """Analyze all betting aspects"""
        # Extract series for analysis
        spreads = [r['spread'] for r in parsed_data if r['spread'] is not None]
        totals = [r['total'] for r in parsed_data if r['total'] is not None]
        ml_away_series = [r['ml_away'] for r in parsed_data if r['ml_away'] is not None]
        ml_home_series = [r['ml_home'] for r in parsed_data if r['ml_home'] is not None]
        
        # Calculate model probabilities
        spread_model_prob = self._calculate_model_probability(spreads) if spreads else 0.5
        total_model_prob = self._calculate_model_probability(totals) if totals else 0.5
        ml_model_prob = self._calculate_model_probability(ml_away_series) if ml_away_series else 0.5
        
        # Calculate EV for each bet type
        spread_ev = self._calculate_ev(spread_model_prob, parsed_data[-1]['spread_vig']) if parsed_data and parsed_data[-1]['spread_vig'] is not None else None
        total_ev = self._calculate_ev(total_model_prob, parsed_data[-1]['total_vig']) if parsed_data and parsed_data[-1]['total_vig'] is not None else None
        ml_ev = self._calculate_ev(ml_model_prob, parsed_data[-1]['ml_away']) if parsed_data and parsed_data[-1]['ml_away'] is not None else None
        
        # Calculate Kelly Criterion
        spread_kelly = self._calculate_kelly(spread_model_prob, parsed_data[-1]['spread_vig']) if parsed_data and parsed_data[-1]['spread_vig'] is not None else None
        total_kelly = self._calculate_kelly(total_model_prob, parsed_data[-1]['total_vig']) if parsed_data and parsed_data[-1]['total_vig'] is not None else None
        ml_kelly = self._calculate_kelly(ml_model_prob, parsed_data[-1]['ml_away']) if parsed_data and parsed_data[-1]['ml_away'] is not None else None
        
        return {
            'spread': {
                'model_prob': spread_model_prob,
                'ev': spread_ev,
                'kelly': spread_kelly
            },
            'total': {
                'model_prob': total_model_prob,
                'ev': total_ev,
                'kelly': total_kelly
            },
            'moneyline': {
                'model_prob': ml_model_prob,
                'ev': ml_ev,
                'kelly': ml_kelly
            },
            'data': parsed_data
        }
    
    def calculate_no_vig_probability(self, odds1: str, odds2: str) -> Dict[str, float]:
        """Calculate no-vig probabilities from two odds values"""
        prob1 = self.implied_probability(odds1)
        prob2 = self.implied_probability(odds2)
        
        # Remove vig
        total_probability = prob1 + prob2
        if total_probability == 0:
            return {'prob1': 0.5, 'prob2': 0.5}
        
        return {
            'prob1': prob1 / total_probability,
            'prob2': prob2 / total_probability
        }
    
    def implied_probability(self, odds: str) -> float:
        """Calculate implied probability from odds"""
        if odds == 'even':
            return 0.5
        if isinstance(odds, str):
            try:
                odds = float(odds)
            except ValueError:
                return 0.5
        if odds > 0:
            return 100 / (odds + 100)
        elif odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 0.5
    
    def _calculate_ev(self, probability: float, odds: str, bet_type: str = 'american') -> float:
        """Calculate expected value for a bet"""
        if bet_type == 'american':
            if odds > 0:
                decimal_odds = 1 + (odds / 100)
            else:
                decimal_odds = 1 + (100 / abs(odds))
        else:
            decimal_odds = float(odds)
        
        # EV = (Probability of Win * Potential Profit) - (Probability of Loss * Stake)
        # Since stake is 1 unit, we simplify:
        return (probability * (decimal_odds - 1)) - ((1 - probability) * 1)
    
    def _calculate_kelly(self, probability: float, odds: str) -> float:
        """Calculate Kelly Criterion stake"""
        if odds > 0:
            decimal_odds = 1 + (odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(odds))
        
        # Kelly % = (BP - Q) / B
        # Where:
        # B = decimal odds - 1
        # P = probability of winning
        # Q = probability of losing (1 - P)
        B = decimal_odds - 1
        P = probability
        Q = 1 - P
        
        return (B * P - Q) / B
    
    def _calculate_model_probability(self, series: List) -> float:
        """Calculate model probability from series (simplified)"""
        if not series:
            return 0.5
        
        # Simple approach: average of last few values
        if len(series) >= 3:
            recent = series[-3:]
            return sum(recent) / len(recent)
        else:
            return sum(series) / len(series) if series else 0.5
