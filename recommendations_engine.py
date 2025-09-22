import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import textwrap
from enum import Enum

class BetType(Enum):
    SPREAD = "spread"
    TOTAL = "total" 
    MONEYLINE = "moneyline"
    PARLAY = "parlay"

class BetConfidence(Enum):
    HIGH = "High Confidence"
    MEDIUM = "Medium Confidence" 
    LOW = "Low Confidence"
    SPECULATIVE = "Speculative"

@dataclass
class BetRecommendation:
    bet_type: BetType
    selection: str
    confidence: BetConfidence
    probability: float
    stake_suggestion: str
    reasoning: List[str]
    key_indicators: Dict[str, float]
    optimal_timing: str
    risk_level: str

class RecommendationsEngine:
    def __init__(self):
        self.betting_terminology = self._initialize_terminology()
        self.market_context = {}
        
    def _initialize_terminology(self) -> Dict[str, List[str]]:
        """Initialize professional betting terminology"""
        return {
            'momentum_phrases': [
                "sharp money pouring in", "steady accumulation", "whale activity detected",
                "reverse line movement", "steam move building", "contrarian indicator flashing"
            ],
            'technical_phrases': [
                "key resistance level", "support holding strong", "breakout confirmation",
                "consolidation pattern", "trend exhaustion", "momentum divergence"
            ],
            'market_phrases': [
                "public heavy on", "sharp action against", "line value present",
                "vig indicates", "market consensus forming", "oddsmaker adjustment"
            ],
            'risk_phrases': [
                "bankroll builder", "cautious approach", "aggressive stance",
                "hedging opportunity", "correlation play", "portfolio diversifier"
            ]
        }
    
    def generate_comprehensive_recommendations(self, ta_signals: List[Any], 
                                            parsed_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate complete betting recommendations across all bet types"""
        
        # Analyze each bet type
        spread_rec = self._analyze_spread_bet(ta_signals, parsed_data)
        total_rec = self._analyze_total_bet(ta_signals, parsed_data) 
        ml_rec = self._analyze_moneylines(ta_signals, parsed_data)
        
        # Determine top pick
        top_pick = self._select_top_pick([spread_rec, total_rec, ml_rec])
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            [spread_rec, total_rec, ml_rec], top_pick, parsed_data
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'executive_summary': executive_summary,
            'top_pick': top_pick,
            'detailed_recommendations': {
                'spread': spread_rec,
                'total': total_rec,
                'moneyline': ml_rec
            },
            'market_context': self._analyze_market_context(parsed_data),
            'betting_strategy': self._generate_betting_strategy([spread_rec, total_rec, ml_rec])
        }
    
    def _analyze_spread_bet(self, ta_signals: List[Any], data: pd.DataFrame) -> BetRecommendation:
        """Generate spread betting recommendation with detailed analysis"""
        
        # Extract latest spread data
        latest = data.iloc[-1] if not data.empty else None
        if latest is None:
            return self._create_default_recommendation(BetType.SPREAD)
        
        # Analyze spread movement and vig
        spread_trend = self._calculate_spread_trend(data['spread'])
        vig_analysis = self._analyze_vig_dynamics(data)
        market_sentiment = self._assess_spread_sentiment(data)
        
        # Determine recommendation
        selection, confidence, probability = self._determine_spread_side(
            latest, spread_trend, vig_analysis, market_sentiment
        )
        
        # Build reasoning
        reasoning = self._build_spread_reasoning(
            latest, spread_trend, vig_analysis, market_sentiment
        )
        
        # Key indicators
        indicators = {
            'current_spread': latest['spread'],
            'vig_discrepancy': vig_analysis.get('discrepancy', 0),
            'trend_strength': spread_trend.get('strength', 0),
            'market_lean': market_sentiment.get('direction', 0)
        }
        
        return BetRecommendation(
            bet_type=BetType.SPREAD,
            selection=selection,
            confidence=confidence,
            probability=probability,
            stake_suggestion=self._suggest_stake_size(confidence, probability),
            reasoning=reasoning,
            key_indicators=indicators,
            optimal_timing=self._determine_optimal_timing(data),
            risk_level=self._assess_risk_level(confidence, probability)
        )
    
    def _analyze_total_bet(self, ta_signals: List[Any], data: pd.DataFrame) -> BetRecommendation:
        """Generate total betting recommendation with over/under analysis"""
        
        latest = data.iloc[-1] if not data.empty else None
        if latest is None:
            return self._create_default_recommendation(BetType.TOTAL)
        
        # Analyze total movement and market pressure
        total_trend = self._calculate_total_trend(data['total'])
        pressure_analysis = self._analyze_total_pressure(data)
        volatility_assessment = self._assess_total_volatility(data)
        
        # Determine over/under recommendation
        selection, confidence, probability = self._determine_total_side(
            latest, total_trend, pressure_analysis, volatility_assessment
        )
        
        reasoning = self._build_total_reasoning(
            latest, total_trend, pressure_analysis, volatility_assessment
        )
        
        indicators = {
            'current_total': latest['total'],
            'pressure_index': pressure_analysis.get('pressure', 0),
            'volatility_score': volatility_assessment.get('score', 0),
            'trend_consistency': total_trend.get('consistency', 0)
        }
        
        return BetRecommendation(
            bet_type=BetType.TOTAL,
            selection=selection,
            confidence=confidence,
            probability=probability,
            stake_suggestion=self._suggest_stake_size(confidence, probability),
            reasoning=reasoning,
            key_indicators=indicators,
            optimal_timing=self._determine_optimal_timing(data, is_total=True),
            risk_level=self._assess_risk_level(confidence, probability)
        )
    
    def _analyze_moneylines(self, ta_signals: List[Any], data: pd.DataFrame) -> BetRecommendation:
        """Generate moneyline betting recommendations"""
        
        latest = data.iloc[-1] if not data.empty else None
        if latest is None:
            return self._create_default_recommendation(BetType.MONEYLINE)
        
        # Analyze ML value and probabilities
        ml_analysis = self._analyze_moneyline_value(data)
        probability_gap = self._calculate_probability_gap(data)
        sharp_money_indicators = self._detect_sharp_money(data)
        
        selection, confidence, probability = self._determine_moneyline_side(
            latest, ml_analysis, probability_gap, sharp_money_indicators
        )
        
        reasoning = self._build_moneyline_reasoning(
            latest, ml_analysis, probability_gap, sharp_money_indicators
        )
        
        indicators = {
            'away_ml_implied': latest.get('away_ml', 0),
            'home_ml_implied': latest.get('home_ml', 0),
            'value_gap': probability_gap.get('gap', 0),
            'sharp_indicator': sharp_money_indicators.get('confidence', 0)
        }
        
        return BetRecommendation(
            bet_type=BetType.MONEYLINE,
            selection=selection,
            confidence=confidence,
            probability=probability,
            stake_suggestion=self._suggest_stake_size(confidence, probability),
            reasoning=reasoning,
            key_indicators=indicators,
            optimal_timing=self._determine_optimal_timing(data),
            risk_level=self._assess_risk_level(confidence, probability)
        )
    
    def _calculate_spread_trend(self, spread_data: pd.Series) -> Dict[str, Any]:
        """Analyze spread movement trends"""
        if len(spread_data) < 5:
            return {'direction': 'neutral', 'strength': 0, 'consistency': 0}
        
        recent = spread_data.tail(10)
        slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
        
        # Calculate trend strength and consistency
        changes = recent.diff().dropna()
        positive_changes = len(changes[changes > 0])
        consistency = abs(positive_changes / len(changes) - 0.5) * 2
        
        return {
            'direction': 'up' if slope > 0.05 else 'down' if slope < -0.05 else 'sideways',
            'strength': abs(slope) * 10,
            'consistency': consistency,
            'recent_movement': recent.iloc[-1] - recent.iloc[0]
        }
    
    def _analyze_vig_dynamics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze vig movements and discrepancies"""
        if len(data) < 2:
            return {'discrepancy': 0, 'movement': 0, 'efficiency': 0}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Calculate vig discrepancy between sides
        fav_vig = latest.get('spread_vig_fav', 0.5)
        dog_vig = latest.get('spread_vig_dog', 0.5)
        discrepancy = abs(fav_vig - dog_vig)
        
        # Vig movement analysis
        vig_movement = abs(latest.get('spread_vig_fav', 0.5) - previous.get('spread_vig_fav', 0.5))
        
        return {
            'discrepancy': discrepancy,
            'movement': vig_movement,
            'efficiency': 1 - discrepancy,  # Higher efficiency = more balanced market
            'fav_vig': fav_vig,
            'dog_vig': dog_vig
        }
    
    def _assess_spread_sentiment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess market sentiment for spread betting"""
        if len(data) < 3:
            return {'direction': 'neutral', 'strength': 0, 'confidence': 0}
        
        # Analyze recent movements for sentiment
        recent = data.tail(5)
        spread_changes = recent['spread'].diff().dropna()
        
        bullish_moves = len(spread_changes[spread_changes < 0])  # Negative spreads becoming more negative
        bearish_moves = len(spread_changes[spread_changes > 0])
        
        sentiment_strength = abs(bullish_moves - bearish_moves) / len(spread_changes)
        
        return {
            'direction': 'bullish' if bullish_moves > bearish_moves else 'bearish',
            'strength': sentiment_strength,
            'confidence': min(sentiment_strength * 2, 1.0),
            'recent_bias': 'favorite' if bullish_moves > bearish_moves else 'underdog'
        }
    
    def _determine_spread_side(self, latest: pd.Series, trend: Dict, 
                             vig_analysis: Dict, sentiment: Dict) -> Tuple[str, BetConfidence, float]:
        """Determine spread betting side with confidence and probability"""
        
        current_spread = latest['spread']
        trend_direction = trend['direction']
        sentiment_direction = sentiment['direction']
        
        # Base probability calculation
        base_prob = 0.5
        adjustments = 0.0
        
        # Trend adjustments
        if trend_direction == 'up':
            adjustments += 0.15 if current_spread < 0 else -0.15  # Favorite getting stronger
        elif trend_direction == 'down':
            adjustments += 0.15 if current_spread > 0 else -0.15  # Underdog getting stronger
            
        # Sentiment adjustments
        if sentiment_direction == 'bullish':
            adjustments += 0.1 if current_spread < 0 else -0.1
        elif sentiment_direction == 'bearish':
            adjustments += 0.1 if current_spread > 0 else -0.1
            
        # Vig efficiency adjustment
        vig_adjustment = vig_analysis.get('efficiency', 0.5) - 0.5
        adjustments += vig_adjustment * 0.1
        
        final_probability = max(0.4, min(0.9, base_prob + adjustments))
        
        # Determine side
        if current_spread < 0:  # Favorite
            side = f"Favorite {current_spread}"
            if adjustments > 0:
                confidence = BetConfidence.HIGH if final_probability > 0.65 else BetConfidence.MEDIUM
            else:
                confidence = BetConfidence.LOW if final_probability < 0.55 else BetConfidence.MEDIUM
        else:  # Underdog
            side = f"Underdog +{abs(current_spread)}"
            if adjustments > 0:
                confidence = BetConfidence.HIGH if final_probability > 0.65 else BetConfidence.MEDIUM
            else:
                confidence = BetConfidence.LOW if final_probability < 0.55 else BetConfidence.MEDIUM
                
        return side, confidence, final_probability
    
    def _build_spread_reasoning(self, latest: pd.Series, trend: Dict, 
                              vig_analysis: Dict, sentiment: Dict) -> List[str]:
        """Build professional reasoning for spread recommendation"""
        
        reasoning = []
        current_spread = latest['spread']
        
        # Trend reasoning
        if trend['strength'] > 0.3:
            if trend['direction'] == 'up':
                reasoning.append(f"Strong upward trend in spread movement (strength: {trend['strength']:.2f})")
            elif trend['direction'] == 'down':
                reasoning.append(f"Pronounced downward trend in line movement (strength: {trend['strength']:.2f})")
        
        # Vig analysis reasoning
        if vig_analysis['discrepancy'] > 0.1:
            reasoning.append(f"Significant vig discrepancy indicating market imbalance")
        elif vig_analysis['efficiency'] > 0.9:
            reasoning.append(f"Highly efficient vig pricing suggests market consensus")
            
        # Sentiment reasoning
        if sentiment['strength'] > 0.6:
            reasoning.append(f"Clear {sentiment['direction']} sentiment detected in recent movement")
            
        # Market context reasoning
        if abs(current_spread) <= 3:
            reasoning.append("Key number spread range - increased volatility expected")
        elif abs(current_spread) >= 7:
            reasoning.append("Wide spread indicates clear market favorite")
            
        return reasoning if reasoning else ["Market showing neutral characteristics - monitoring for breakout"]
    
    def _analyze_total_pressure(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze over/under pressure from market data"""
        if len(data) < 2:
            return {'pressure': 0, 'direction': 'neutral', 'consistency': 0}
        
        latest = data.iloc[-1]
        
        # Analyze prefix for pressure direction
        prefix = latest.get('total_prefix', '').lower()
        over_vig = latest.get('total_vig_over', 0.5)
        under_vig = latest.get('total_vig_under', 0.5)
        
        pressure_direction = 1 if prefix == 'o' else -1 if prefix == 'u' else 0
        pressure_strength = abs(over_vig - under_vig)
        
        return {
            'pressure': pressure_strength * pressure_direction,
            'direction': 'over' if pressure_direction > 0 else 'under' if pressure_direction < 0 else 'balanced',
            'consistency': pressure_strength,
            'current_prefix': prefix
        }
    
    def _build_total_reasoning(self, latest: pd.Series, trend: Dict, 
                             pressure: Dict, volatility: Dict) -> List[str]:
        """Build professional reasoning for total recommendation"""
        
        reasoning = []
        current_total = latest['total']
        prefix = latest.get('total_prefix', '')
        
        # Pressure reasoning
        if abs(pressure['pressure']) > 0.1:
            direction = "over" if pressure['pressure'] > 0 else "under"
            reasoning.append(f"Strong {direction} pressure detected (prefix: {prefix})")
            
        # Volatility reasoning
        if volatility['score'] > 0.7:
            reasoning.append("High volatility environment favors aggressive total positioning")
        elif volatility['score'] < 0.3:
            reasoning.append("Low volatility conditions suggest cautious approach")
            
        # Key number reasoning
        if 44 <= current_total <= 48:  # Common NFL total range
            reasoning.append("Total in high-frequency scoring range")
        elif current_total >= 52:
            reasoning.append("Elevated total suggests offensive matchup")
            
        return reasoning if reasoning else ["Total market showing balanced characteristics"]
    
    def _select_top_pick(self, recommendations: List[BetRecommendation]) -> Dict[str, Any]:
        """Select the strongest recommendation as top pick"""
        if not recommendations:
            return {'selection': 'No clear top pick', 'reasoning': ['Insufficient data']}
        
        # Score each recommendation
        scored_recs = []
        for rec in recommendations:
            score = self._score_recommendation(rec)
            scored_recs.append((score, rec))
        
        # Select highest score
        scored_recs.sort(key=lambda x: x[0], reverse=True)
        top_rec = scored_recs[0][1]
        
        return {
            'bet_type': top_rec.bet_type.value,
            'selection': top_rec.selection,
            'confidence': top_rec.confidence.value,
            'probability': top_rec.probability,
            'reasoning': top_rec.reasoning,
            'why_top_pick': self._explain_top_pick(top_rec, scored_recs)
        }
    
    def _score_recommendation(self, recommendation: BetRecommendation) -> float:
        """Score recommendation based on confidence and probability"""
        confidence_scores = {
            BetConfidence.HIGH: 1.0,
            BetConfidence.MEDIUM: 0.7, 
            BetConfidence.LOW: 0.4,
            BetConfidence.SPECULATIVE: 0.2
        }
        
        base_score = confidence_scores.get(recommendation.confidence, 0.5)
        probability_boost = (recommendation.probability - 0.5) * 0.5
        
        return base_score + probability_boost
    
    def _explain_top_pick(self, top_rec: BetRecommendation, 
                         all_recs: List[Tuple[float, BetRecommendation]]) -> List[str]:
        """Explain why this is the top pick"""
        reasons = []
        
        # Confidence-based reasoning
        if top_rec.confidence == BetConfidence.HIGH:
            reasons.append("Highest confidence level among all recommendations")
        elif top_rec.confidence == BetConfidence.MEDIUM:
            reasons.append("Strongest signal in current market conditions")
            
        # Probability-based reasoning
        if top_rec.probability > 0.65:
            reasons.append("Exceptionally high probability estimate")
        elif top_rec.probability > 0.55:
            reasons.append("Solid probability edge over market")
            
        # Comparative reasoning
        if len(all_recs) > 1:
            next_best_score = all_recs[1][0] if len(all_recs) > 1 else 0
            score_gap = all_recs[0][0] - next_best_score
            if score_gap > 0.2:
                reasons.append("Significantly stronger signal than alternatives")
                
        return reasons
    
    def _generate_executive_summary(self, recommendations: List[BetRecommendation],
                                  top_pick: Dict, data: pd.DataFrame) -> str:
        """Generate professional executive summary"""
        
        summary_parts = []
        
        # Market overview
        market_context = self._analyze_market_context(data)
        summary_parts.append(f"Market Analysis: {market_context['overview']}")
        
        # Top pick highlight
        summary_parts.append(
            f"Top Recommendation: {top_pick['selection']} "
            f"({top_pick['bet_type'].upper()}) - {top_pick['confidence']} "
            f"with {top_pick['probability']:.1%} probability"
        )
        
        # Strategy insight
        strategy = self._generate_betting_strategy(recommendations)
        summary_parts.append(f"Betting Strategy: {strategy['primary_approach']}")
        
        # Risk assessment
        risk_levels = [rec.risk_level for rec in recommendations]
        avg_risk = max(set(risk_levels), key=risk_levels.count)
        summary_parts.append(f"Overall Risk Profile: {avg_risk}")
        
        return "\n".join(summary_parts)
    
    def _analyze_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market context"""
        if data.empty:
            return {'overview': 'Limited market data available', 'volatility': 'unknown'}
        
        # Calculate market volatility
        if len(data) >= 5:
            spread_volatility = data['spread'].std()
            total_volatility = data['total'].std() if 'total' in data.columns else 0
            
            volatility_assessment = 'High' if spread_volatility > 1.0 else 'Moderate' if spread_volatility > 0.5 else 'Low'
        else:
            volatility_assessment = 'Unknown'
            
        return {
            'overview': f"Market showing {volatility_assessment.lower()} volatility conditions",
            'volatility': volatility_assessment,
            'data_points': len(data),
            'time_range': f"{data['timestamp'].min()} to {data['timestamp'].max()}" if 'timestamp' in data.columns else 'Unknown'
        }
    
    def _generate_betting_strategy(self, recommendations: List[BetRecommendation]) -> Dict[str, str]:
        """Generate overall betting strategy based on recommendations"""
        
        high_confidence_count = sum(1 for rec in recommendations if rec.confidence == BetConfidence.HIGH)
        avg_probability = np.mean([rec.probability for rec in recommendations])
        
        if high_confidence_count >= 2 and avg_probability > 0.6:
            approach = "Aggressive positioning recommended with multiple high-confidence opportunities"
        elif high_confidence_count >= 1:
            approach = "Selective aggression on top picks with cautious approach on others"
        else:
            approach = "Conservative approach advised - focus on bankroll preservation"
            
        return {
            'primary_approach': approach,
            'recommended_actions': self._generate_recommended_actions(recommendations),
            'risk_considerations': self._generate_risk_considerations(recommendations)
        }
    
    def _generate_recommended_actions(self, recommendations: List[BetRecommendation]) -> List[str]:
        """Generate specific betting actions"""
        actions = []
        
        for rec in recommendations:
            if rec.confidence == BetConfidence.HIGH:
                actions.append(f"Go heavy on {rec.bet_type.value}: {rec.selection}")
            elif rec.confidence == BetConfidence.MEDIUM:
                actions.append(f"Standard position on {rec.bet_type.value}: {rec.selection}")
            else:
                actions.append(f"Light position or pass on {rec.bet_type.value}")
                
        return actions
    
    def _generate_risk_considerations(self, recommendations: List[BetRecommendation]) -> List[str]:
        """Generate risk management considerations"""
        considerations = []
        
        high_risk_count = sum(1 for rec in rec.risk_level == 'High' for rec in recommendations)
        if high_risk_count >= 2:
            considerations.append("Multiple high-risk positions - consider correlation hedging")
            
        if any(rec.probability < 0.45 for rec in recommendations):
            considerations.append("Some recommendations below 50% probability - ensure proper bankroll management")
            
        return considerations if considerations else ["Standard risk management protocols apply"]
    
    def _suggest_stake_size(self, confidence: BetConfidence, probability: float) -> str:
        """Suggest appropriate stake size"""
        base_units = {
            BetConfidence.HIGH: 3,
            BetConfidence.MEDIUM: 2,
            BetConfidence.LOW: 1,
            BetConfidence.SPECULATIVE: 0.5
        }
        
        units = base_units.get(confidence, 1)
        # Adjust for probability
        if probability > 0.7:
            units *= 1.5
        elif probability < 0.5:
            units *= 0.7
            
        return f"{units:.1f} units (1 unit = 1% of bankroll)"
    
    def _assess_risk_level(self, confidence: BetConfidence, probability: float) -> str:
        """Assess risk level for recommendation"""
        if confidence == BetConfidence.HIGH and probability > 0.65:
            return "Low"
        elif confidence == BetConfidence.MEDIUM and probability > 0.55:
            return "Medium"
        else:
            return "High"
    
    def _determine_optimal_timing(self, data: pd.DataFrame, is_total: bool = False) -> str:
        """Determine optimal betting timing"""
        if len(data) < 10:
            return "Monitor for additional data"
            
        recent_volatility = data.iloc[-5:].std().mean()
        
        if recent_volatility > 1.0:
            return "Wait for volatility to settle"
        elif is_total:
            return "Place 2-4 hours before game time"
        else:
            return "Ideal timing: 1-2 hours before kickoff"
    
    def _create_default_recommendation(self, bet_type: BetType) -> BetRecommendation:
        """Create default recommendation when data is insufficient"""
        return BetRecommendation(
            bet_type=bet_type,
            selection="No recommendation - insufficient data",
            confidence=BetConfidence.LOW,
            probability=0.5,
            stake_suggestion="0 units",
            reasoning=["Awaiting additional market data"],
            key_indicators={},
            optimal_timing="Monitor market",
            risk_level="Unknown"
        )

# Example usage with the TA engine output
def demonstrate_recommendations_engine(ta_signals: List[Any], parsed_data: pd.DataFrame) -> Dict[str, Any]:
    """Demonstrate the recommendations engine with sample data"""
    
    engine = RecommendationsEngine()
    
    # Generate comprehensive recommendations
    recommendations = engine.generate_comprehensive_recommendations(ta_signals, parsed_data)
    
    # Display results in professional format
    print("=" * 80)
    print("PROFESSIONAL BETTING RECOMMENDATIONS ENGINE")
    print("=" * 80)
    print(f"\n📊 EXECUTIVE SUMMARY")
    print("-" * 40)
    print(recommendations['executive_summary'])
    
    print(f"\n🎯 TOP PICK")
    print("-" * 40)
    top_pick = recommendations['top_pick']
    print(f"Bet Type: {top_pick['bet_type'].upper()}")
    print(f"Selection: {top_pick['selection']}")
    print(f"Confidence: {top_pick['confidence']}")
    print(f"Probability: {top_pick['probability']:.1%}")
    print(f"Why Top Pick: {', '.join(top_pick['why_top_pick'])}")
    
    print(f"\n📈 DETAILED RECOMMENDATIONS")
    print("-" * 40)
    
    for bet_type, rec in recommendations['detailed_recommendations'].items():
        print(f"\n{bet_type.upper()} BET:")
        print(f"  Selection: {rec.selection}")
        print(f"  Confidence: {rec.confidence.value}")
        print(f"  Probability: {rec.probability:.1%}")
        print(f"  Stake: {rec.stake_suggestion}")
        print(f"  Risk Level: {rec.risk_level}")
        print(f"  Optimal Timing: {rec.optimal_timing}")
        print(f"  Key Reasoning:")
        for reason in rec.reasoning:
            print(f"    • {reason}")
    
    print(f"\n⚡ BETTING STRATEGY")
    print("-" * 40)
    strategy = recommendations['betting_strategy']
    print(f"Primary Approach: {strategy['primary_approach']}")
    print(f"Recommended Actions:")
    for action in strategy['recommended_actions']:
        print(f"  • {action}")
    
    return recommendations

# Sample data simulation for testing
def create_sample_data() -> pd.DataFrame:
    """Create sample parsed data for testing"""
    dates = pd.date_range(start='2024-01-15', periods=20, freq='H')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'spread': np.random.normal(-3.5, 0.5, 20),
        'total': np.random.normal(47.5, 1.0, 20),
        'spread_vig_fav': np.random.uniform(0.52, 0.58, 20),
        'spread_vig_dog': np.random.uniform(0.48, 0.52, 20),
        'total_vig_over': np.random.uniform(0.50, 0.55, 20),
        'total_vig_under': np.random.uniform(0.50, 0.55, 20),
        'away_ml': np.random.uniform(0.45, 0.55, 20),
        'home_ml': np.random.uniform(0.45, 0.55, 20),
        'total_prefix': ['o' if x > 0.5 else 'u' for x in np.random.random(20)]
    })
    
    return sample_data

if __name__ == "__main__":
    # Test with sample data
    sample_ta_signals = []  # Would normally come from TA engine
    sample_parsed_data = create_sample_data()
    
    recommendations = demonstrate_recommendations_engine(sample_ta_signals, sample_parsed_data)
    
    # Save to file for review
    import json
    with open('betting_recommendations.json', 'w') as f:
        # Convert to serializable format
        serializable_recs = json.loads(json.dumps(recommendations, default=str))
        json.dump(serializable_recs, f, indent=2)
    
    print(f"\n✅ Full recommendations saved to 'betting_recommendations.json'")
