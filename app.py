import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional, Any
import math

# Import our custom modules
from ta_engine import TechnicalAnalysisEngine
from betting_analyzer import BettingAnalyzer
from recommendations_engine import RecommendationsEngine
from backtester import Backtester

# Initialize engines
ta_engine = TechnicalAnalysisEngine()
betting_analyzer = BettingAnalyzer()
recommendations_engine = RecommendationsEngine()
backtester = Backtester()

# Streamlit app
st.set_page_config(
    page_title="Omniscience - Enhanced TA Engine",
    page_icon="📊",
    layout="wide"
)

st.title("Omniscience — Enhanced TA Engine (EV + Kelly + Backtesting)")
st.markdown("""
    **Professional Sports Betting Analysis Tool**
    
    Paste odds feed in the format:
    - Line 1: Time [Team] Spread
    - Line 2: Spread Vig
    - Line 3: Total (e.g. O 154.5 / U 154.5)
    - Line 4: Total Vig
    - Line 5: Away ML Home ML
""")

# Bankroll management
bankroll = st.number_input("Bankroll ($)", value=1000.0, min_value=1.0, step=100.0)

# Data input
st.subheader("Odds Feed Input")
raw_data = st.text_area(
    "Paste odds data here (first line should be header)",
    height=300,
    placeholder="time 10/15 12:00PM\nLAC -3.5\n-110\nO 154.5\n-110\n-120 105\n\n"
)

# Process data
if st.button("Analyze"):
    if not raw_data.strip():
        st.error("Please paste odds data first.")
    else:
        # Parse the data
        try:
            parsed_data = parse_blocks_strict(raw_data)
            if not parsed_data:
                st.error("No valid data blocks found. Ensure first line is header and each block has 5 lines.")
            else:
                # Analyze with all engines
                analysis_result = analyze_with_all_engines(parsed_data, bankroll)
                
                # Display results
                st.subheader("Analysis Results")
                st.markdown(analysis_result['html'])
                
                # Display recommendation
                st.subheader("Professional Recommendation")
                st.markdown(f"<div style='background: linear-gradient(90deg, #0a2c3d, #082230); padding: 15px; border-radius: 8px; border-left: 4px solid #7ee3d0;'><h3 style='color: #7ee3d0; margin-top: 0;'>{analysis_result['rec']}</h3><p style='color: #e6f7f6;'>Confidence: {analysis_result['conf']}</p></div>", unsafe_allow_html=True)
                
                # Display parsed preview
                st.subheader("Parsed Data Preview")
                preview_df = pd.DataFrame([
                    {
                        'Time': row['time'].strftime('%m/%d %H:%M'),
                        'Team': row['team'] or '',
                        'Spread': row['spread'],
                        'SpreadVig': row['spread_vig'],
                        'Total': row['total'],
                        'TotalVig': row['total_vig'],
                        'AwayML': row['ml_away'],
                        'HomeML': row['ml_home'],
                        'AwayML Prob': f"{row['ml_away_prob']*100:.1f}%" if row['ml_away_prob'] is not None else '',
                        'HomeML Prob': f"{row['ml_home_prob']*100:.1f}%" if row['ml_home_prob'] is not None else ''
                    }
                    for row in parsed_data
                ])
                st.dataframe(preview_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

# Backtesting
if st.button("Run Backtest"):
    if not raw_data.strip():
        st.error("Please paste odds data first.")
    else:
        try:
            parsed_data = parse_blocks_strict(raw_data)
            if not parsed_data:
                st.error("No valid data blocks found.")
            else:
                backtest_result = backtester.run_backtest(parsed_data, bankroll)
                st.subheader("Backtest Results")
                st.markdown(backtest_result['html'])
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")

# Function to parse blocks
def parse_blocks_strict(raw: str) -> List[Dict]:
    """Parse raw odds data into structured format"""
    lines = [line.strip() for line in raw.split('\n') if line.strip()]
    start = 1 if is_header_line(lines[0]) else 0
    rows = []
    errors = []

    for i in range(start, len(lines) - 4, 5):
        try:
            block_lines = lines[i:i+5]
            if len(block_lines) < 5:
                break
                
            L1, L2, L3, L4, L5 = block_lines
            
            # Parse timestamp
            tks = L1.split()
            if len(tks) < 2:
                errors.append({
                    'index': i,
                    'reason': 'Invalid timestamp format',
                    'raw': block_lines
                })
                continue
                
            date_token = tks[0]
            time_token = tks[1]
            full_timestamp = f"{date_token} {time_token}"
            
            time = parse_timestamp(full_timestamp)
            if time is None:
                errors.append({
                    'index': i,
                    'reason': 'Invalid timestamp',
                    'raw': block_lines
                })
                continue

            # Team and spread extraction
            team = None
            spread_raw = None
            
            if len(tks) >= 3:
                if has_letters(tks[2]):
                    team = tks[2]
                    spread_raw = tks[3] if len(tks) > 3 else None
                else:
                    spread_raw = tks[2]
            
            spread = normalize_spread(spread_raw) if spread_raw else None

            # Spread vig - extract first number and calculate opposite
            spread_vig = extract_first_number(L2)
            spread_vig_opposite = calculate_opposite_vig(spread_vig) if spread_vig is not None else None

            # Total parsing
            total_side = None
            total = None
            
            m = re.match(r'^([ouOU])\s*([+-]?\d+(?:\.\d+)?)', L3)
            if m:
                total_side = m.group(1).lower()
                total = float(m.group(2))
            else:
                total = extract_first_number(L3)
                if L3 and L3[0].lower() in ['o', 'u']:
                    total_side = L3[0].lower()

            # Total vig - extract first number and calculate opposite
            total_vig = extract_first_number(L4)
            total_vig_opposite = calculate_opposite_vig(total_vig) if total_vig is not None else None

            # Moneylines parsing
            ml_nums = extract_numbers(L5)
            ml_away = None
            ml_home = None
            
            if 'even' in L5.lower():
                ml_away = 'even'
                remaining = L5.split('even')[1]
                ml_home = extract_first_number(remaining)
            else:
                if len(ml_nums) >= 1:
                    ml_away = ml_nums[0]
                if len(ml_nums) >= 2:
                    ml_home = ml_nums[1]

            # Calculate no-vig probabilities
            spread_no_vig = None
            total_no_vig = None
            ml_no_vig = None
            
            if spread_vig is not None and spread_vig_opposite is not None:
                spread_no_vig = betting_analyzer.calculate_no_vig_probability(str(spread_vig), str(spread_vig_opposite))
            
            if total_vig is not None and total_vig_opposite is not None:
                total_no_vig = betting_analyzer.calculate_no_vig_probability(str(total_vig), str(total_vig_opposite))
            
            if ml_away is not None and ml_home is not None:
                ml_no_vig = betting_analyzer.calculate_no_vig_probability(ml_away, str(ml_home))

            # Store all values
            rows.append({
                'time': time,
                'team': team,
                'spread': spread,
                'spread_vig': spread_vig,
                'spread_vig_opposite': spread_vig_opposite,
                'spread_no_vig': spread_no_vig,
                'total': total,
                'total_side': total_side,
                'total_vig': total_vig,
                'total_vig_opposite': total_vig_opposite,
                'total_no_vig': total_no_vig,
                'ml_away': ml_away,
                'ml_home': ml_home,
                'ml_no_vig': ml_no_vig,
                'raw': block_lines,
                'ml_away_prob': betting_analyzer.implied_probability(ml_away) if ml_away is not None else None,
                'ml_home_prob': betting_analyzer.implied_probability(ml_home) if ml_home is not None else None
            })
            
        except Exception as e:
            errors.append({
                'index': i,
                'reason': f'Parsing error: {str(e)}',
                'raw': block_lines
            })
            continue
    
    # Sort by time
    rows.sort(key=lambda x: x['time'])
    return rows

# Helper functions
def is_header_line(line: str) -> bool:
    """Check if line is a header line"""
    return line.lower().startswith('time')

def has_letters(s: str) -> bool:
    """Check if string contains letters"""
    return bool(re.search(r'[A-Za-z]', s))

def normalize_spread(spread_str: str) -> Optional[float]:
    """Normalize spread value, preserving directional meaning"""
    try:
        if spread_str == 'even':
            return 100.0  # Convert even to +100 for TA purposes
        val = float(spread_str)
        # Return as-is since we want to preserve the directional meaning
        return val
    except ValueError:
        return None

def extract_numbers(s: str) -> List[float]:
    """Extract all numbers from a string"""
    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', s)
    return [float(x) for x in numbers]

def extract_first_number(s: str) -> Optional[float]:
    """Extract first number from a string"""
    numbers = extract_numbers(s)
    return numbers[0] if numbers else None

def parse_timestamp(time_str: str) -> Optional[datetime]:
    """Parse timestamp in MM/DD h:mmAM/PM format"""
    try:
        # Handle both formats: MM/DD h:mmAM/PM and MM/DD h:mm
        if 'AM' in time_str or 'PM' in time_str:
            # MM/DD h:mmAM/PM format
            parts = time_str.split()
            if len(parts) < 2:
                return None
            date_part = parts[0]
            time_part = parts[1]
            # Parse date
            month, day = map(int, date_part.split('/'))
            # Parse time
            time_only, period = time_part[:-2], time_part[-2:]
            hour, minute = map(int, time_only.split(':'))
            
            if period.upper() == 'PM' and hour < 12:
                hour += 12
            elif period.upper() == 'AM' and hour == 12:
                hour = 0
                
            return datetime(2000, month, day, hour, minute, 0, 0)
        else:
            # MM/DD h:mm format
            parts = time_str.split()
            if len(parts) < 2:
                return None
            date_part = parts[0]
            time_part = parts[1]
            month, day = map(int, date_part.split('/'))
            hour, minute = map(int, time_part.split(':'))
            return datetime(2000, month, day, hour, minute, 0, 0)
    except Exception:
        return None

def calculate_opposite_vig(vig: float) -> float:
    """Calculate opposite vig using standard dime line system"""
    # Standard practice: if one side is -110, the other is typically -110
    # This is a simplification but works for most cases
    return vig

# Main analysis function
def analyze_with_all_engines(parsed_data: List[Dict], bankroll: float) -> Dict[str, Any]:
    """Run all analysis engines on parsed data"""
    if not parsed_data:
        return {
            'html': '<div class="analysis-block" style="color:#ffdcdc">No valid blocks parsed.</div>',
            'rec': 'No data',
            'conf': 'Low'
        }
    
    # Run technical analysis
    ta_results = ta_engine.analyze_all(parsed_data)
    
    # Run betting analysis
    betting_results = betting_analyzer.analyze_all(parsed_data)
    
    # Run recommendations
    recommendations = recommendations_engine.generate_recommendations(
        ta_results, 
        betting_results, 
        bankroll
    )
    
    # Combine results
    html = recommendations['html']
    
    return {
        'html': html,
        'rec': recommendations['rec'],
        'conf': recommendations['conf']
    }

# Add custom CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #2dd4bf;
        color: #042024;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #28b8a8;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame th {
        background-color: #0f3b3a;
        color: #e6f7f6;
    }
    .stDataFrame td {
        background-color: rgba(255,255,255,0.03);
    }
</style>
""", unsafe_allow_html=True)
