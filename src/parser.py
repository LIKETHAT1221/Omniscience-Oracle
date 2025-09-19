import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Optional, Union
from config.settings import config

class OmniscienceDataParser:
    def __init__(self):
        self.parsed_data = []
        self._time_series_data = {}
        
    def parse_main_blocks(self, main_data_text: str) -> List[Dict]:
        """Parse main odds feed with toggle support"""
        lines = main_data_text.strip().split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            if config.ignore_header_rows and self._is_header_line(stripped):
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                continue
                
            current_block.append(stripped)
        
        if current_block:
            blocks.append(current_block)
            
        results = []
        for block in blocks:
            try:
                if len(block) == 5 and config.parse_5_line_blocks:
                    parsed = self._parse_5_line_block(block)
                elif len(block) == 4 and config.parse_4_line_blocks:
                    parsed = self._parse_4_line_block(block)
                else:
                    continue
                    
                results.append(parsed)
                self.parsed_data.append(parsed)
            except Exception as e:
                print(f"Error parsing block: {e}")
                continue
                
        return results
    
    from typing import Dict, List
import numpy as np
from config.settings import config

class RecommendationEngine:
    def __init__(self):
        self.confidence_thresholds = {
            'strong': config.strong_confidence_threshold,
            'moderate': config.min_confidence_for_action,
            'weak': config.min_confidence_for_action - 0.1
        }
        
    def generate_recommendation(self, game_data: Dict, ta_indicators: Dict) -> Dict:
        """Generate recommendation using multi-indicator pipeline"""
        confidence = 0.5
        narrative = []
        triggered_indicators = []
        
        # RLM Analysis
        rlm_strength = self._calculate_rlm_strength(game_data)
        if rlm_strength is not None and abs(rlm_strength) > 10:
            narrative.append(self._generate_rlm_narrative(game_data, rlm_strength))
            confidence += 0.2 if rlm_strength > 0 else 0.15
            triggered_indicators.append('rlm')
        
        # Multi-indicator analysis pipeline
        indicator_analysis = self._analyze_ta_indicators(ta_indicators)
        confidence += indicator_analysis['confidence_impact']
        narrative.extend(indicator_analysis['narrative'])
        triggered_indicators.extend(indicator_analysis['triggered_indicators'])
        
        # Generate final recommendation
        recommendation, strength = self._determine_recommendation(confidence, indicator_analysis)
        
        # Calculate sizing and expected value
        expected_value = self._calculate_expected_value(confidence, indicator_analysis)
        kelly_stake = self._calculate_kelly_stake(expected_value)
        
        return {
            'market_type': game_data.get('game_type', 'unknown'),
            'recommendation': f"{strength}{recommendation}",
            'confidence': confidence,
            'narrative': " ".join(narrative),
            'expected_value': expected_value,
            'kelly_stake': kelly_stake,
            'timestamp': game_data.get('parsed_at', ''),
            'triggered_indicators': triggered_indicators,
            'indicator_summary': indicator_analysis['summary']
        }
    
    def _analyze_ta_indicators(self, ta_indicators: Dict) -> Dict:
        """Analyze all TA indicators based on config toggles"""
        confidence_impact = 0.0
        narrative = []
        triggered_indicators = []
        indicator_summary = {}
        
        for market, indicators in ta_indicators.items():
            market_narrative = []
            market_confidence = 0.0
            
            # Momentum analysis
            if config.use_momentum_indicators and 'momentum' in indicators:
                mom_analysis = self._analyze_momentum(indicators['momentum'], market)
                market_confidence += mom_analysis['confidence_impact']
                market_narrative.append(mom_analysis['narrative'])
                if mom_analysis['triggered']:
                    triggered_indicators.append(f'{market}_momentum')
            
            # RSI analysis
            if config.use_rsi and 'rsi' in indicators:
                rsi_analysis = self._analyze_rsi(indicators['rsi'], market)
                market_confidence += rsi_analysis['confidence_impact']
                market_narrative.append(rsi_analysis['narrative'])
                if rsi_analysis['triggered']:
                    triggered_indicators.append(f'{market}_rsi')
            
            # ... [similar blocks for all other indicators] ...
            
            if market_narrative:
                narrative.extend(market_narrative)
                confidence_impact += market_confidence
                indicator_summary[market] = {
                    'confidence_impact': market_confidence,
                    'indicators_triggered': [ind for ind in triggered_indicators if ind.startswith(market)]
                }
        
        return {
            'confidence_impact': confidence_impact,
            'narrative': narrative,
            'triggered_indicators': triggered_indicators,
            'summary': indicator_summary
        }
    
    # ... [individual indicator analysis methods] ...# ... [rest of your existing parser methods] ...
