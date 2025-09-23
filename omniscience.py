# omniscience.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

# ---------------------------
# Custom modules (handles both standalone and package mode)
# ---------------------------
try:
    # Standalone mode (files in the same folder)
    from line_movement import track_line_movement
    from backtester import Backtester
    from ta_engine import TechnicalAnalysisEngine
    from betting_analyzer import BettingAnalyzer
    from recommendations_engine import RecommendationsEngine
    from odds_utils import american_to_prob, no_vig_prob, american_to_decimal, decimal_to_american
except ImportError:
    # Package mode (folder treated as a package)
    from .line_movement import track_line_movement
    from .backtester import Backtester
    from .ta_engine import TechnicalAnalysisEngine
    from .betting_analyzer import BettingAnalyzer
    from .recommendations_engine import RecommendationsEngine
    from .odds_utils import american_to_prob, no_vig_prob, american_to_decimal, decimal_to_american

# ---------------------------
# Sports-betting package
# ---------------------------
from sportsbetting import implied_probability, convert_odds

# ---------------------------
# Initialize engines
# ---------------------------
ta_engine = TechnicalAnalysisEngine()
betting_analyzer = BettingAnalyzer()
recommendations_engine = RecommendationsEngine()
backtester = Backtester()

# ---------------------------
# SQLite setup
# ---------------------------
conn = sqlite3.connect("omniscience.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    raw_data TEXT,
    rec TEXT,
    conf TEXT
)
""")
conn.commit()

# ---------------------------
# Streamlit setup
# ---------------------------
st.set_page_config(page_title="Omniscience - Enhanced TA Engine", page_icon="📊", layout="wide")
st.title("Omniscience — Enhanced TA Engine (EV + Kelly + Backtesting)")

st.markdown("""
**Professional Sports Betting Analysis Tool**

Paste odds feed in the format:
- Line 1: Time [Team] Spread
- Line 2: Spread Vig
- Line 3: Total (O/U)
- Line 4: Total Vig
- Line 5: Away ML Home ML
""")

# ---------------------------
# Bankroll input
# ---------------------------
bankroll = st.number_input("Bankroll ($)", value=1000.0, min_value=1.0, step=100.0)

# ---------------------------
# Odds feed input
# ---------------------------
st.subheader("Odds Feed Input")
raw_data = st.text_area(
    "Paste odds data here (first line should be header)",
    height=300,
    placeholder="time 10/15 12:00PM\nLAC -3.5\n-110\nO 154.5\n-110\n-120 105\n\n"
)

# ---------------------------
# Parsing helpers
# ---------------------------
def is_header_line(line: str) -> bool:
    return line.lower().startswith("time")

def parse_blocks_strict(raw: str) -> List[Dict]:
    """
    Parse raw odds data into structured format.
    """
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    start = 1 if is_header_line(lines[0]) else 0
    rows = []

    for i in range(start, len(lines)-4, 5):
        block = lines[i:i+5]
        try:
            L1, L2, L3, L4, L5 = block
            # Timestamp
            date_part, time_part, *team_tokens = L1.split()
            month, day = map(int, date_part.split("/"))
            hour, minute = map(int, time_part[:-2].split(":"))
            period = time_part[-2:].upper()
            if period == "PM" and hour < 12: hour += 12
            if period == "AM" and hour == 12: hour = 0
            time = datetime(2000, month, day, hour, minute)
            # Team & spread
            team = team_tokens[0] if team_tokens else None
            spread = float(team_tokens[1]) if len(team_tokens) > 1 else None
            spread_vig = float(L2)
            total_side = L3[0].lower() if L3[0].lower() in ["o","u"] else None
            total = float(L3.split()[1])
            total_vig = float(L4)
            # Moneylines
            ml_tokens = L5.split()
            ml_away, ml_home = None, None
            for t in ml_tokens:
                if t.lower() == "even": t = "+100"
                if ml_away is None: ml_away = float(t)
                else: ml_home = float(t)
            # No-vig probabilities using sports-betting
            spread_no_vig = implied_probability(spread_vig)
            total_no_vig = implied_probability(total_vig)
            ml_no_vig = (implied_probability(ml_away), implied_probability(ml_home)) if ml_away and ml_home else None

            rows.append({
                "time": time, "team": team, "spread": spread, "spread_vig": spread_vig,
                "total": total, "total_side": total_side, "total_vig": total_vig,
                "ml_away": ml_away, "ml_home": ml_home,
                "spread_no_vig": spread_no_vig, "total_no_vig": total_no_vig, "ml_no_vig": ml_no_vig
            })
        except Exception as e:
            st.error(f"Error parsing block at line {i}: {e}")
            continue
    return rows

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("Analyze"):
    if not raw_data.strip():
        st.error("Please paste odds data first.")
    else:
        parsed_data = parse_blocks_strict(raw_data)
        # Track line movement for spreads and totals (5-period smoothing)
        parsed_data = track_line_movement(parsed_data, "spread", smooth_window=5)
        parsed_data = track_line_movement(parsed_data, "total", smooth_window=5)
        # Run engines
        ta_results = ta_engine.analyze_all(parsed_data)
        betting_results = betting_analyzer.analyze_all(parsed_data)
        recommendations = recommendations_engine.generate_recommendations(ta_results, betting_results, bankroll)
        # Store results in SQLite
        cursor.execute(
            "INSERT INTO analysis_results (timestamp, raw_data, rec, conf) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), raw_data, recommendations['rec'], recommendations['conf'])
        )
        conn.commit()
        # Display results
        st.subheader("Analysis Results")
        st.markdown(recommendations['html'], unsafe_allow_html=True)
        st.subheader("Professional Recommendation")
        st.markdown(
            f"<div style='background: linear-gradient(90deg,#0a2c3d,#082230);padding:15px;border-radius:8px;border-left:4px solid #7ee3d0;'>"
            f"<h3 style='color:#7ee3d0;margin-top:0'>{recommendations['rec']}</h3>"
            f"<p style='color:#e6f7f6'>Confidence: {recommendations['conf']}</p></div>",
            unsafe_allow_html=True
        )
        # Display parsed preview with momentum
        st.subheader("Parsed Data with Line Movement")
        preview_df = pd.DataFrame([
            {
                'Time': row['time'].strftime('%m/%d %H:%M'),
                'Team': row.get('team'),
                'Spread': row.get('spread'),
                'Spread Δ': row.get('spread_delta'),
                'Spread Momentum': row.get('spread_momentum'),
                'Spread Momentum (Smooth)': row.get('spread_momentum_smooth'),
                'Total': row.get('total'),
                'Total Δ': row.get('total_delta'),
                'Total Momentum': row.get('total_momentum'),
                'Total Momentum (Smooth)': row.get('total_momentum_smooth'),
                'Away ML': row.get('ml_away'),
                'Home ML': row.get('ml_home')
            }
            for row in parsed_data
        ])
        def highlight_momentum(val):
            if pd.isna(val): return ''
            elif val > 0: return 'color:#2dd4bf;font-weight:bold;'
            elif val < 0: return 'color:#f87171;font-weight:bold;'
            else: return ''
        st.dataframe(preview_df.style.applymap(highlight_momentum, subset=[
            'Spread Δ','Spread Momentum','Spread Momentum (Smooth)',
            'Total Δ','Total Momentum','Total Momentum (Smooth)'
        ]), use_container_width=True)

# ---------------------------
# Backtesting Button
# ---------------------------
if st.button("Run Backtest"):
    if not raw_data.strip():
        st.error("Please paste odds data first.")
    else:
        parsed_data = parse_blocks_strict(raw_data)
        backtest_result = backtester.run_backtest(parsed_data, bankroll)
        st.subheader("Backtest Results")
        st.markdown(backtest_result['html'], unsafe_allow_html=True)
