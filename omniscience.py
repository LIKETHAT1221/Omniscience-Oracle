# omniscience_core.py
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple

class OmniscienceDataParser:
    """
    The core parser for the Omniscience system.
    Handles main odds blocks (4-line, 5-line) and optional splits blocks (8-line).
    All odds are immediately converted to Implied Probability.
    """
    def __init__(self):
        self.parsed_data = []  # Stores all parsed game data
        self._time_series_data = {}  # For storing historical data by game key
        
    def parse_main_blocks(self, main_data_text: str) -> List[Dict]:
        """Parse the main odds feed (4-line or 5-line blocks)."""
        lines = main_data_text.strip().split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Check if this is a header line
            if self._is_header_line(stripped):
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
                if len(block) == 5:
                    parsed = self._parse_5_line_block(block)
                elif len(block) == 4:
                    parsed = self._parse_4_line_block(block)
                else:
                    print(f"Skipping block with unexpected length {len(block)}: {block}")
                    continue
                    
                results.append(parsed)
                self.parsed_data.append(parsed)
            except Exception as e:
                print(f"Error parsing block: {e}\nBlock content: {block}")
                
        return results
        
    def parse_splits_blocks(self, splits_data_text: str) -> List[Dict]:
        """Parse the splits data feed (8-line blocks)."""
        lines = splits_data_text.strip().split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            current_block.append(stripped)
            if len(current_block) == 8:
                blocks.append(current_block)
                current_block = []
        
        results = []
        for block in blocks:
            try:
                parsed = self._parse_splits_block(block)
                results.append(parsed)
            except Exception as e:
                print(f"Error parsing splits block: {e}\nBlock content: {block}")
                
        return results
        
    def merge_data(self, main_data: List[Dict], splits_data: List[Dict]) -> List[Dict]:
        """Merge splits data into the main data based on order."""
        for i, game_data in enumerate(main_data):
            if i < len(splits_data):
                game_data.update(splits_data[i])
        return main_data
        
    def _is_header_line(self, line: str) -> bool:
        """Determine if a line is a header that should be ignored."""
        header_indicators = ['time', 'spread', 'total', 'ml', 'runline', 'open']
        lower_line = line.lower()
        return any(indicator in lower_line for indicator in header_indicators)
        
    def _parse_5_line_block(self, block: List[str]) -> Dict:
        """Parse a 5-line spread game block."""
        # Line 1: Date/Time and Spread info (e.g., "9/17 12:56PM BUF -12.5")
        line1_parts = block[0].split()
        if len(line1_parts) < 4:
            raise ValueError(f"Line 1 should have at least 4 parts: {block[0]}")
            
        datetime_str = f"{line1_parts[0]} {line1_parts[1]}"
        
        # The favorite team name might consist of multiple tokens
        spread_index = -1  # The spread is the last token
        favorite_team = " ".join(line1_parts[2:spread_index])
        spread = line1_parts[spread_index]
        
        # Line 2: spread vig (e.g., "-112")
        spread_vig = block[1].strip()
        
        # Line 3: total with o/u prefix (e.g., "o49.5")
        total_line = block[2].strip()
        
        # Line 4: total vig (e.g., "-112")
        total_vig = block[3].strip()
        
        # Line 5: moneyline values (e.g., "+390 -850")
        ml_parts = block[4].split()
        if len(ml_parts) < 2:
            ml_parts = block[4].split('\t')
            
        if len(ml_parts) != 2:
            raise ValueError(f"Invalid moneyline format: {block[4]}")
            
        away_ml, home_ml = ml_parts
        
        # Create a dictionary with the parsed data
        parsed = {
            'game_type': 'spread',
            'datetime': datetime_str,
            'favorite_team': favorite_team,
            'spread': spread,
            'spread_vig': spread_vig,
            'total': total_line,
            'total_vig': total_vig,
            'away_ml': away_ml,
            'home_ml': home_ml,
            'parsed_at': datetime.now().isoformat(),
            # Initialize splits fields to None
            'away_bets_pct': None,
            'home_bets_pct': None,
            'away_money_pct': None,
            'home_money_pct': None
        }
        
        # Convert all odds to implied probabilities
        self._convert_to_implied_probabilities(parsed)
        
        return parsed
        
    def _parse_4_line_block(self, block: List[str]) -> Dict:
        """Parse a 4-line moneyline game block."""
        # Line 1: Time, SD ML, NYM ML, Total, Runline (e.g., "9/18 9:01AM +121 -147 u9")
        line1_parts = block[0].split()
        if len(line1_parts) < 5:
            raise ValueError(f"Line 1 should have at least 5 parts: {block[0]}")
            
        datetime_str = f"{line1_parts[0]} {line1_parts[1]}"
        away_ml = line1_parts[2]
        home_ml = line1_parts[3]
        total = line1_parts[4]
        
        # Line 2: total vig (e.g., "-112")
        total_vig = block[1].strip()
        
        # Line 3: runline info (e.g., "NYM -1.5")
        runline_info = block[2].strip()
        runline_parts = runline_info.split()
        if len(runline_parts) < 2:
            raise ValueError(f"Invalid runline format: {runline_info}")
            
        runline_team = runline_parts[0]
        runline = runline_parts[1]
        
        # Line 4: runline vig (e.g., "+149")
        runline_vig = block[3].strip()
        
        # Create a dictionary with the parsed data
        parsed = {
            'game_type': 'moneyline',
            'datetime': datetime_str,
            'away_ml': away_ml,
            'home_ml': home_ml,
            'total': total,
            'total_vig': total_vig,
            'runline_team': runline_team,
            'runline': runline,
            'runline_vig': runline_vig,
            'parsed_at': datetime.now().isoformat(),
            # Initialize splits fields to None
            'away_bets_pct': None,
            'home_bets_pct': None,
            'away_money_pct': None,
            'home_money_pct': None
        }
        
        # Convert all odds to implied probabilities
        self._convert_to_implied_probabilities(parsed)
        
        return parsed
        
    def _parse_splits_block(self, block: List[str]) -> Dict:
        """Parse an 8-line splits block. Lines 4-7 contain the key data."""
        if len(block) != 8:
            raise ValueError(f"Splits block must have exactly 8 lines, got {len(block)}")
            
        # Lines 4-7: away bets %, home bets %, away money %, home money %
        away_bets = block[3].strip().replace('%', '')
        home_bets = block[4].strip().replace('%', '')
        away_money = block[5].strip().replace('%', '')
        home_money = block[6].strip().replace('%', '')
        
        try:
            return {
                'away_bets_pct': float(away_bets),
                'home_bets_pct': float(home_bets),
                'away_money_pct': float(away_money),
                'home_money_pct': float(home_money)
            }
        except ValueError as e:
            raise ValueError(f"Error parsing splits data: {e}")
            
    def _convert_to_implied_probabilities(self, data: Dict) -> None:
        """Convert all odds in the data to implied probabilities."""
        # Convert moneyline odds
        if 'away_ml' in data and data['away_ml']:
            data['away_ml_implied'] = self._odds_to_implied_probability(data['away_ml'])
        if 'home_ml' in data and data['home_ml']:
            data['home_ml_implied'] = self._odds_to_implied_probability(data['home_ml'])
            
        # Convert vig odds
        if 'spread_vig' in data and data['spread_vig']:
            data['spread_vig_implied'] = self._odds_to_implied_probability(data['spread_vig'])
        if 'total_vig' in data and data['total_vig']:
            data['total_vig_implied'] = self._odds_to_implied_probability(data['total_vig'])
        if 'runline_vig' in data and data['runline_vig']:
            data['runline_vig_implied'] = self._odds_to_implied_probability(data['runline_vig'])
            
    def _odds_to_implied_probability(self, odds_str: str) -> float:
        """Convert American odds string to implied probability."""
        if odds_str.lower() == 'even':
            return 0.5
            
        try:
            odds = float(odds_str)
        except ValueError:
            return None
            
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
            
    def add_to_time_series(self, game_data: Dict) -> None:
        """Add game data to the time series storage."""
        # Create a unique key for this game
        if game_data['game_type'] == 'spread':
            game_key = f"{game_data['favorite_team']}_{game_data['datetime']}"
        else:
            game_key = f"{game_data['away_ml']}_{game_data['home_ml']}_{game_data['datetime']}"
            
        if game_key not in self._time_series_data:
            self._time_series_data[game_key] = []
            
        self._time_series_data[game_key].append(game_data)
        
    def get_time_series(self, game_key: str) -> List[Dict]:
        """Get time series data for a specific game."""
        return self._time_series_data.get(game_key, [])
        
    def calculate_ta_indicators(self, game_key: str) -> Dict:
        """Calculate TA indicators for a game's time series data."""
        series_data = self.get_time_series(game_key)
        if not series_data:
            return {}
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(series_data)
        
        # Sort by parsed_at timestamp
        df['parsed_at'] = pd.to_datetime(df['parsed_at'])
        df = df.sort_values('parsed_at')
        
        # Calculate indicators for each market
        results = {}
        
        # Calculate for away ML if available
        if 'away_ml_implied' in df.columns and df['away_ml_implied'].notna().any():
            away_ml_series = df['away_ml_implied'].values
            results['away_ml'] = calculate_momentum_indicators(away_ml_series)
            
        # Calculate for home ML if available
        if 'home_ml_implied' in df.columns and df['home_ml_implied'].notna().any():
            home_ml_series = df['home_ml_implied'].values
            results['home_ml'] = calculate_momentum_indicators(home_ml_series)
            
        # Calculate for spread vig if available
        if 'spread_vig_implied' in df.columns and df['spread_vig_implied'].notna().any():
            spread_vig_series = df['spread_vig_implied'].values
            results['spread_vig'] = calculate_momentum_indicators(spread_vig_series)
            
        return results


# --- TA Stack Core Functions ---
def calculate_momentum_indicators(series: np.array, period: int = 2) -> Dict[str, float]:
    """
    Calculates Momentum Velocity (1st derivative) and
    Momentum Acceleration (2nd derivative) for a series.
    """
    if len(series) < period + 1:
        return {'MOM_V': None, 'MOM_A': None}
        
    # Calculate Momentum Velocity (1st derivative)
    mom_v = (series[-1] - series[-period-1]) / period
    
    # Calculate Momentum Acceleration (2nd derivative)
    if len(series) >= 2 * period + 1:
        prev_mom_v = (series[-period-1] - series[-2*period-1]) / period
        mom_a = (mom_v - prev_mom_v) / period
    else:
        mom_a = None
        
    return {'MOM_V': mom_v, 'MOM_A': mom_a}


def calculate_rsi(series: np.array, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI) for a series.
    """
    if len(series) < period + 1:
        return None
        
    # Calculate price changes
    deltas = np.diff(series)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gains = np.mean(gains[-period:])
    avg_losses = np.mean(losses[-period:])
    
    # Avoid division by zero
    if avg_losses == 0:
        return 100
        
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# --- Recommendation Engine ---
class RecommendationEngine:
    """Generates FADE/BACK/HOLD recommendations based on TA indicators and market data."""
    
    def __init__(self):
        pass
        
    def generate_recommendation(self, game_data: Dict, ta_indicators: Dict) -> Dict:
        """
        Generate a recommendation for a game based on current data and TA indicators.
        Returns: {'market_type', 'recommendation', 'confidence', 'narrative', 'expected_value', 'kelly_stake'}
        """
        # Base confidence and narrative
        confidence = 0.5
        narrative = []
        
        # Check for RLM if splits data is available
        if (game_data.get('away_bets_pct') is not None and 
            game_data.get('away_money_pct') is not None):
            
            away_bets = game_data['away_bets_pct']
            home_bets = game_data['home_bets_pct']
            away_money = game_data['away_money_pct']
            home_money = game_data['home_money_pct']
            
            # Determine which side has more bets (public sentiment)
            if away_bets > home_bets:
                public_side = 'away'
                public_pct = away_bets
                sharp_pct = home_money
            else:
                public_side = 'home'
                public_pct = home_bets
                sharp_pct = away_money
            
            # Calculate RLM strength
            rlm_strength = sharp_pct - (100 - public_pct)
            
            if abs(rlm_strength) > 10:  # Significant RLM
                narrative.append(
                    f"RLM detected: {public_pct:.0f}% bets on {public_side}, "
                    f"but {sharp_pct:.0f}% money on opposite side."
                )
                
                if rlm_strength > 0:
                    # Sharp money is on opposite side of public - FADE opportunity
                    confidence += 0.3
                    recommendation = "FADE"
                else:
                    # Sharp money is on same side as public - caution
                    confidence += 0.1
                    recommendation = "BACK"
            else:
                narrative.append("No significant RLM detected.")
                recommendation = "HOLD"
        else:
            narrative.append("No splits data available.")
            recommendation = "HOLD"
        
        # Add TA indicators to narrative if available
        for market, indicators in ta_indicators.items():
            if indicators['MOM_V'] is not None:
                narrative.append(
                    f"{market}: MOM_V={indicators['MOM_V']:.4f}, "
                    f"MOM_A={indicators['MOM_A']:.4f if indicators['MOM_A'] is not None else 'N/A'}"
                )
                
                # Use momentum to adjust confidence
                if indicators['MOM_V'] > 0 and indicators['MOM_A'] > 0:
                    confidence += 0.1
                elif indicators['MOM_V'] < 0 and indicators['MOM_A'] < 0:
                    confidence -= 0.1
        
        # Determine recommendation strength
        if confidence >= 0.7:
            strength = "STRONG "
        elif confidence >= 0.6:
            strength = ""
        else:
            strength = ""
            recommendation = "HOLD"  # Override to HOLD if confidence is too low
        
        # Calculate expected value and Kelly stake (simplified for now)
        expected_value = max(0, (confidence - 0.5) * 0.1)  # Placeholder
        kelly_stake = min(0.05, expected_value * 2)  # Conservative Kelly
        
        return {
            'market_type': game_data.get('game_type', 'unknown'),
            'recommendation': f"{strength}{recommendation}",
            'confidence': confidence,
            'narrative': " ".join(narrative),
            'expected_value': expected_value,
            'kelly_stake': kelly_stake
        }


# --- Streamlit App ---
def create_streamlit_app():
    """Create and run the Streamlit app for Omniscience."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Please install it with: pip install streamlit")
        return
        
    st.title("Omniscience Sports Betting Analysis")
    
    # Initialize parser and recommendation engine
    if 'parser' not in st.session_state:
        st.session_state.parser = OmniscienceDataParser()
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()
    
    # Input for main odds blocks
    st.header("Input Odds Data")
    main_data_text = st.text_area("Paste main odds blocks (4-line or 5-line):", height=200)
    
    # Input for splits blocks
    splits_data_text = st.text_area("Paste splits blocks (8-line, optional):", height=200)
    
    if st.button("Analyze"):
        if main_data_text:
            with st.spinner("Parsing and analyzing data..."):
                # Parse main data
                main_data = st.session_state.parser.parse_main_blocks(main_data_text)
                
                # Parse splits data if provided
                splits_data = []
                if splits_data_text:
                    splits_data = st.session_state.parser.parse_splits_blocks(splits_data_text)
                
                # Merge data
                merged_data = st.session_state.parser.merge_data(main_data, splits_data)
                
                # Add to time series and calculate TA indicators
                recommendations = []
                for game in merged_data:
                    # Add to time series
                    st.session_state.parser.add_to_time_series(game)
                    
                    # Create a game key
                    if game['game_type'] == 'spread':
                        game_key = f"{game['favorite_team']}_{game['datetime']}"
                    else:
                        game_key = f"{game['away_ml']}_{game['home_ml']}_{game['datetime']}"
                    
                    # Calculate TA indicators
                    ta_indicators = st.session_state.parser.calculate_ta_indicators(game_key)
                    
                    # Generate recommendation
                    recommendation = st.session_state.recommendation_engine.generate_recommendation(
                        game, ta_indicators
                    )
                    recommendations.append(recommendation)
                
                # Display results
                st.header("Analysis Results")
                
                for i, rec in enumerate(recommendations):
                    st.subheader(f"Game {i+1} - {rec['market_type'].upper()}")
                    st.write(f"**Recommendation:** {rec['recommendation']}")
                    st.write(f"**Confidence:** {rec['confidence']:.2%}")
                    st.write(f"**Expected Value:** {rec['expected_value']:.2%}")
                    st.write(f"**Kelly Stake:** {rec['kelly_stake']:.2%}")
                    st.write(f"**Narrative:** {rec['narrative']}")
                    st.write("---")
        else:
            st.warning("Please enter some odds data to analyze.")


if __name__ == "__main__":
    create_streamlit_app()
