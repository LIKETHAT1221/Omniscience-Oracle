import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from parser import OmniscienceDataParser
from recommendation_engine import RecommendationEngine
from ta_engine import calculate_ta_indicators

def main():
    st.set_page_config(
        page_title="Omniscience Oracle",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔮 Omniscience Oracle - Advanced Sports Betting Analysis")
    st.markdown("---")
    
    if 'parser' not in st.session_state:
        st.session_state.parser = OmniscienceDataParser()
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    
    tab1, tab2, tab3 = st.tabs(["Live Analysis", "Historical Data", "Configuration"])
    
    with tab1:
        st.header("Real-time Market Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Data")
            main_data = st.text_area(
                "Paste Main Odds Blocks (4-line or 5-line format):",
                height=200,
                help="Paste the main odds data in 4-line or 5-line format"
            )
            
            splits_data = st.text_area(
                "Paste Splits Blocks (8-line format, optional):",
                height=150,
                help="Paste the splits data in 8-line format (optional)"
            )
            
            if st.button("Analyze Markets", type="primary", use_container_width=True):
                if main_data:
                    analyze_markets(main_data, splits_data)
                else:
                    st.warning("Please enter main odds data to analyze.")
        
        with col2:
            st.subheader("Analysis Results")
            
            if st.session_state.analysis_results:
                for i, result in enumerate(st.session_state.analysis_results):
                    with st.expander(f"Game {i+1} - {result['market_type'].upper()}", expanded=i==0):
                        display_recommendation(result)
            else:
                st.info("Enter data and click 'Analyze Markets' to see recommendations")
    
    with tab2:
        st.header("Historical Analysis & Backtesting")
        st.info("Historical analysis features will be implemented in the next version")
    
    with tab3:
        st.header("System Configuration")
        st.info("Configuration options will be implemented in the next version")

def analyze_markets(main_data: str, splits_data: str = None):
    with st.spinner("Analyzing market data..."):
        try:
            main_results = st.session_state.parser.parse_main_blocks(main_data)
            
            splits_results = []
            if splits_data:
                splits_results = st.session_state.parser.parse_splits_blocks(splits_data)
            
            merged_data = st.session_state.parser.merge_data(main_results, splits_results)
            
            recommendations = []
            for game in merged_data:
                game_key = st.session_state.parser.add_to_time_series(game)
                
                time_series_data = st.session_state.parser.get_time_series(game_key)
                
                ta_indicators = {}
                
                if game['game_type'] == 'spread':
                    markets = ['away_ml_implied', 'home_ml_implied', 'spread_vig_implied', 'total_vig_implied']
                else:
                    markets = ['away_ml_implied', 'home_ml_implied', 'total_vig_implied', 'runline_vig_implied']
                
                for market in markets:
                    if any(market in data_point for data_point in time_series_data):
                        ta_indicators[market] = calculate_ta_indicators(time_series_data, market)
                
                recommendation = st.session_state.recommendation_engine.generate_recommendation(
                    game, ta_indicators
                )
                recommendations.append(recommendation)
            
            st.session_state.analysis_results = recommendations
            st.success(f"Analysis complete! Generated {len(recommendations)} recommendations.")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.exception(e)

def display_recommendation(recommendation: Dict):
    if "STRONG" in recommendation['recommendation']:
        st.success(f"**{recommendation['recommendation']}**")
    elif "BACK" in recommendation['recommendation']:
        st.info(f"**{recommendation['recommendation']}**")
    elif "FADE" in recommendation['recommendation']:
        st.warning(f"**{recommendation['recommendation']}**")
    else:
        st.write(f"**{recommendation['recommendation']}**")
    
    st.progress(recommendation['confidence'])
    st.caption(f"Confidence: {recommendation['confidence']:.0%}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Value", f"{recommendation['expected_value']:+.2%}")
    with col2:
        st.metric("Recommended Stake", f"{recommendation['kelly_stake']:.1%}")
    with col3:
        st.metric("Market Type", recommendation['market_type'])
    
    st.subheader("Analysis Details")
    st.write(recommendation['narrative'])
    
    st.caption(f"Analysis performed at: {recommendation['timestamp']}")

if __name__ == "__main__":
    main()
