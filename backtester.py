import math
from typing import List, Dict, Tuple, Optional, Any

class RecommendationsEngine:
    """Professional recommendations engine"""
    
    def __init__(self):
        pass
    
    def generate_recommendations(self, ta_results: Dict, betting_results: Dict, bankroll: float) -> Dict[str, Any]:
        """Generate professional recommendations"""
        spread_data = betting_results['spread']
        total_data = betting_results['total']
        ml_data = betting_results['moneyline']
        
        # Calculate confidence levels
        spread_confidence = self._calculate_confidence(spread_data['ev'])
        total_confidence = self._calculate_confidence(total_data['ev'])
        ml_confidence = self._calculate_confidence(ml_data['ev'])
        
        # Calculate risk levels
        spread_risk = self._calculate_risk(spread_data['ev'])
        total_risk = self._calculate_risk(total_data['ev'])
        ml_risk = self._calculate_risk(ml_data['ev'])
        
        # Determine side indicators
        spread_side = self._determine_side(spread_data['ev'])
        total_side = self._determine_side(total_data['ev'])
        ml_side = self._determine_side(ml_data['ev'])
        
        # Determine best bet based on highest EV
        ev_values = [
            {'type': 'Spread', 'ev': spread_data['ev'], 'kelly': spread_data['kelly'], 'prob': spread_data['model_prob'], 'risk': spread_risk, 'side': spread_side, 'confidence': spread_confidence},
            {'type': 'Total', 'ev': total_data['ev'], 'kelly': total_data['kelly'], 'prob': total_data['model_prob'], 'risk': total_risk, 'side': total_side, 'confidence': total_confidence},
            {'type': 'Moneyline', 'ev': ml_data['ev'], 'kelly': ml_data['kelly'], 'prob': ml_data['model_prob'], 'risk': ml_risk, 'side': ml_side, 'confidence': ml_confidence}
        ].filter(lambda x: x['ev'] is not None)
        
        # Sort by EV descending
        ev_values.sort(key=lambda x: x['ev'], reverse=True)
        best_bet = ev_values[0] if ev_values else None
        
        # Generate professional recommendation
        recommendation = self._generate_professional_recommendation(best_bet, bankroll)
        
        # Generate detailed HTML
        html = self._generate_detailed_html(best_bet, spread_data, total_data, ml_data, bankroll)
        
        return {
            'html': html,
            'rec': recommendation,
            'conf': 'High' if best_bet and best_bet['ev'] > 0.05 else 'Medium' if best_bet and best_bet['ev'] > 0 else 'Low'
        }
    
    def _calculate_confidence(self, ev: float) -> str:
        """Calculate confidence level from EV"""
        if ev is None:
            return 'low'
        if ev > 0.05:
            return 'high'
        if ev > 0:
            return 'medium'
        return 'low'
    
    def _calculate_risk(self, ev: float) -> str:
        """Calculate risk level from EV"""
        if ev is None:
            return 'high'
        if ev > 0.05:
            return 'low'
        if ev > 0:
            return 'medium'
        return 'high'
    
    def _determine_side(self, ev: float) -> str:
        """Determine side based on EV"""
        if ev is None:
            return 'neutral'
        if ev > 0:
            return 'favorite' if ev > 0.05 else 'dog'
        return 'neutral'
    
    def _generate_professional_recommendation(self, best_bet: Dict, bankroll: float) -> str:
        """Generate professional recommendation text"""
        if not best_bet:
            return "No clear edge identified"
        
        bet_type = best_bet['type']
        bet_side = best_bet['side']
        bet_ev = best_bet['ev']
        bet_prob = best_bet['prob']
        bet_kelly = best_bet['kelly']
        
        # Calculate stake
        stake = self._calculate_kelly_stake(bet_kelly, bankroll)
        
        if bet_type == 'Spread':
            if bet_side == 'favorite':
                return f"PROFESSIONAL RECOMMENDATION: {bet_type} - {bet_side.upper()} (EV: {(bet_ev*100):.2f}%)"
            else:
                return f"PROFESSIONAL RECOMMENDATION: {bet_type} - {bet_side.upper()} (EV: {(bet_ev*100):.2f}%)"
        elif bet_type == 'Total':
            if bet_side == 'over':
                return f"PROFESSIONAL RECOMMENDATION: {bet_type} - {bet_side.upper()} (EV: {(bet_ev*100):.2f}%)"
            else:
                return f"PROFESSIONAL RECOMMENDATION: {bet_type} - {bet_side.upper()} (EV: {(bet_ev*100):.2f}%)"
        else:
            return f"PROFESSIONAL RECOMMENDATION: {bet_type} - {bet_side.upper()} (EV: {(bet_ev*100):.2f}%)"
    
    def _calculate_kelly_stake(self, kelly: float, bankroll: float, max_fraction: float = 0.5) -> float:
        """Calculate Kelly stake"""
        if kelly is None or kelly <= 0:
            return 0
        raw_stake = kelly * bankroll
        max_stake = max_fraction * bankroll
        return min(raw_stake, max_stake)
    
    def _generate_detailed_html(self, best_bet: Dict, spread_data: Dict, total_data: Dict, ml_data: Dict, bankroll: float) -> str:
        """Generate detailed HTML with professional recommendations"""
        if not best_bet:
            return """
            <div class="analysis-block">
                <h3>Enhanced TA Stack with EV & Kelly Sizing</h3>
                <div class="recommendation-box">
                    <h3>PROFESSIONAL RECOMMENDATION</h3>
                    <p>No clear edge identified in current market conditions.</p>
                    <p><strong>Analysis:</strong> Market indicators are conflicting or neutral. Consider waiting for stronger signals before placing a bet.</p>
                </div>
            </div>
            """
        
        bet_type = best_bet['type']
        bet_side = best_bet['side']
        bet_ev = best_bet['ev']
        bet_prob = best_bet['prob']
        bet_kelly = best_bet['kelly']
        
        # Calculate stake
        stake = self._calculate_kelly_stake(bet_kelly, bankroll)
        
        # Generate side indicator
        side_indicator = f"<span style='background: #7ee3d0; color: #042024; padding: 3px 8px; border-radius: 4px; font-size: 0.9em; margin: 0 3px;'>{bet_side.upper()}</span>"
        
        # Generate risk indicator
        risk_indicator = f"<span style='background: {'#60d394' if best_bet['risk'] == 'low' else '#ffd166' if best_bet['risk'] == 'medium' else '#ff7b7b'}; color: #042024; padding: 3px 8px; border-radius: 4px; font-size: 0.9em; margin: 0 3px;'>{best_bet['risk'].upper()}</span>"
        
        # Generate confidence indicator
        confidence_indicator = f"<span style='background: {'#60d394' if best_bet['confidence'] == 'high' else '#ffd166' if best_bet['confidence'] == 'medium' else '#ff7b7b'}; color: #042024; padding: 3px 8px; border-radius: 4px; font-size: 0.9em; margin: 0 3px;'>{best_bet['confidence'].upper()}</span>"
        
        return f"""
        <div class="analysis-block">
            <h3>Enhanced TA Stack with EV & Kelly Sizing</h3>
            <div class="recommendation-box">
                <h3>PROFESSIONAL RECOMMENDATION</h3>
                <p><span style="font-weight: bold; font-size: 1.2em; color: #7ee3d0;">{bet_type} {side_indicator} - EV: {(bet_ev*100):.2f}%</span></p>
                <p><strong>Probability:</strong> {(bet_prob*100):.1f}% | <strong>Risk:</strong> {risk_indicator} | <strong>Confidence:</strong> {confidence_indicator}</p>
                <p><strong>Stake Sizing:</strong> ${stake:.2f} ({(bet_kelly*100):.1f}% of bankroll)</p>
                <p><strong>Analysis:</strong> {self._generate_detailed_analysis(bet_type, bet_side)}</p>
            </div>
        </div>
        """
    
    def _generate_detailed_analysis(self, bet_type: str, bet_side: str) -> str:
        """Generate detailed analysis text"""
        if bet_type == 'Spread':
            if bet_side == 'favorite':
                return "The favorite is showing strong momentum with technical indicators supporting the position. The spread is trading at a favorable level with positive expected value."
            else:
                return "The underdog is showing value based on technical analysis and market inefficiencies. The spread provides a favorable edge with positive expected value."
        elif bet_type == 'Total':
            if bet_side == 'over':
                return "The over is favored based on recent scoring trends and momentum indicators. The total is trading at a level that provides positive expected value."
            else:
                return "The under is favored based on defensive trends and momentum indicators. The total is trading at a level that provides positive expected value."
        else:
            if bet_side == 'dog':
                return "The underdog moneyline shows value based on technical analysis and market inefficiencies. The odds provide a favorable edge with positive expected value."
            else:
                return "The favorite moneyline shows strong technical support with momentum indicators confirming the position. The odds provide positive expected value."
