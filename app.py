import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
import os

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(__file__))

# Import our engines
from odds_parser import OddsParser
from ta_engine import AdvancedTechnicalAnalysis, TradingSignal
from recommendations_engine import RecommendationsEngine, BetRecommendation

# Configure the page
st.set_page_config(
    page_title="Omniscience Odds TA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .signal-buy {
        border-color: #28a745;
        background-color: #f8fff9;
    }
    .signal-sell {
        border-color: #dc3545;
        background-color: #fff8f8;
    }
    .signal-hold {
        border-color: #ffc107;
        background-color: #fffef0;
    }
    .top-pick {
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class SportsBettingApp:
    def __init__(self):
        self.parser = OddsParser()
        self.ta_engine = AdvancedTechnicalAnalysis()
        self.rec_engine = RecommendationsEngine()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = pd.DataFrame()
        if 'current_recommendations' not in st.session_state:
            st.session_state.current_recommendations = {}
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
    
    def main(self):
        """Main application interface"""
        st.markdown('<div class="main-header">⚡ Omniscience Odds TA Engine</div>', 
                   unsafe_allow_html=True)
        
        # Sidebar for data input
        with st.sidebar:
            self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_main_content()
        
        with col2:
            self.render_analysis_panel()
    
    def render_sidebar(self):
        """Render the sidebar controls"""
        st.header("📊 Data Input")
        
        # Feed type selection
        feed_type = st.selectbox(
            "Select Feed Type",
            ["5line", "4line", "splits"],
            help="Choose the format of your odds data"
        )
        
        # Game information
        game_name = st.text_input("Game Label", "KC vs BUF")
        sport_type = st.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB"])
        
        # Data input
        st.subheader("Odds Data Input")
        raw_feed = st.text_area(
            "Paste Odds Feed Here",
            height=200,
            help="Paste your 5-line blocks of odds data"
        )
        
        # Example data for testing
        with st.expander("Example Format"):
            st.code("""
HEADER IGNORE
2024-01-15 19:30 KC -3.5
-115
o47.5
-110
+150 -180
2024-01-15 16:00 BUF +2.5
-105
u42.5
-115
+120 -140
            """)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Parse & Analyze", type="primary", use_container_width=True):
                self.process_odds_data(game_name, feed_type, raw_feed, sport_type)
        
        with col2:
            if st.button("🔄 Clear Data", use_container_width=True):
                self.clear_data()
        
        # Data management
        st.subheader("Data Management")
        if st.button("💾 Save Current Analysis", use_container_width=True):
            self.save_analysis()
        
        if st.button("📊 View History", use_container_width=True):
            self.view_analysis_history()
    
    def process_odds_data(self, game_name: str, feed_type: str, raw_feed: str, sport_type: str):
        """Process the odds data through the entire pipeline"""
        try:
            with st.spinner("🔄 Parsing odds data..."):
                # Parse the raw feed
                parsed_data = self.parser.parse_feed(raw_feed)
                
                if parsed_data.empty:
                    st.error("❌ No valid data parsed. Please check your input format.")
                    return
            
            with st.spinner("📈 Running technical analysis..."):
                # Add to historical data
                if st.session_state.historical_data.empty:
                    st.session_state.historical_data = parsed_data
                else:
                    st.session_state.historical_data = pd.concat([
                        st.session_state.historical_data, parsed_data
                    ], ignore_index=True)
                
                # Run TA analysis
                signals = self.ta_engine.generate_trading_signals(
                    current_data=parsed_data.iloc[-1].to_dict() if not parsed_data.empty else {},
                    historical_data=st.session_state.historical_data
                )
            
            with st.spinner("🎯 Generating recommendations..."):
                # Generate recommendations
                recommendations = self.rec_engine.generate_comprehensive_recommendations(
                    signals, st.session_state.historical_data
                )
                
                # Store current recommendations
                st.session_state.current_recommendations = recommendations
                
                # Add to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'game': game_name,
                    'sport': sport_type,
                    'recommendations': recommendations
                })
            
            st.success("✅ Analysis complete!")
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
    
    def render_main_content(self):
        """Render the main content area"""
        if not st.session_state.current_recommendations:
            self.render_welcome_screen()
        else:
            self.render_analysis_results()
    
    def render_welcome_screen(self):
        """Render welcome screen when no data is available"""
        st.markdown("""
        ## 🎯 Welcome to Omniscience Odds TA
        
        This advanced sports betting analysis platform combines:
        
        - **Real-time odds parsing** from multiple formats
        - **Technical analysis** with 15+ indicators
        - **Machine learning** pattern recognition
        - **Professional betting recommendations**
        
        ### 🚀 Getting Started
        
        1. **Select your feed type** in the sidebar
        2. **Enter game information** (name, sport)
        3. **Paste your odds data** in the provided format
        4. **Click 'Parse & Analyze'** to generate recommendations
        
        ### 📊 Supported Analysis
        
        - **Point Spreads**: Favorite/underdog movement tracking
        - **Totals**: Over/under market analysis  
        - **Moneylines**: Probability and value detection
        - **Vig Analysis**: Market efficiency assessment
        
        ### 💡 Pro Tips
        
        - Use consistent timestamp formats for best results
        - Include at least 3-5 data points for meaningful analysis
        - Monitor line movements over time for pattern recognition
        """)
        
        # Quick start example
        with st.expander("🎯 Quick Start Example"):
            st.code("""
# Sample NFL Odds Data
HEADER IGNORE
2024-01-20 20:00 KC -3.5
-110
o47.5
-110
+150 -180
2024-01-20 19:30 BUF +2.5
-105
u45.5
-115
+130 -150
            """)
    
    def render_analysis_results(self):
        """Render the analysis results"""
        recs = st.session_state.current_recommendations
        
        # Top Pick Highlight
        st.markdown('<div class="top-pick">', unsafe_allow_html=True)
        st.subheader("🎯 TOP PICK")
        
        top_pick = recs.get('top_pick', {})
        if top_pick:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Bet Type", top_pick.get('bet_type', '').upper())
                st.metric("Selection", top_pick.get('selection', ''))
            
            with col2:
                st.metric("Confidence", top_pick.get('confidence', ''))
                st.metric("Probability", f"{top_pick.get('probability', 0)*100:.1f}%")
            
            with col3:
                st.metric("Risk Level", "LOW" if top_pick.get('probability', 0) > 0.6 else "MEDIUM")
            
            # Why top pick
            st.write("**Why this pick:**")
            for reason in top_pick.get('why_top_pick', []):
                st.write(f"• {reason}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Executive Summary
        st.subheader("📊 Executive Summary")
        st.write(recs.get('executive_summary', ''))
        
        # Detailed Recommendations
        st.subheader("📈 Detailed Recommendations")
        
        detailed_recs = recs.get('detailed_recommendations', {})
        
        for bet_type, recommendation in detailed_recs.items():
            self.render_recommendation_card(bet_type.upper(), recommendation)
    
    def render_recommendation_card(self, bet_type: str, recommendation: BetRecommendation):
        """Render individual recommendation card"""
        # Determine card style based on confidence
        confidence_class = {
            "High Confidence": "signal-buy",
            "Medium Confidence": "signal-hold", 
            "Low Confidence": "signal-sell"
        }.get(recommendation.confidence.value, "signal-hold")
        
        st.markdown(f'<div class="signal-card {confidence_class}">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{bet_type}**")
            st.write(f"*{recommendation.selection}*")
            
            # Reasoning
            with st.expander("Analysis Reasoning"):
                for reason in recommendation.reasoning:
                    st.write(f"• {reason}")
        
        with col2:
            st.metric("Confidence", recommendation.confidence.value)
            st.metric("Probability", f"{recommendation.probability*100:.1f}%")
        
        with col3:
            st.metric("Stake", recommendation.stake_suggestion)
            st.metric("Risk", recommendation.risk_level)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_analysis_panel(self):
        """Render the right-side analysis panel"""
        st.subheader("📋 Analysis Panel")
        
        if st.session_state.historical_data.empty:
            st.info("No data available. Process some odds data to see analysis.")
            return
        
        # Current Data Summary
        with st.expander("📊 Current Data Summary"):
            data = st.session_state.historical_data
            st.write(f"**Total Data Points:** {len(data)}")
            st.write(f"**Date Range:** {data['date'].min()} to {data['date'].max()}")
            
            # Quick stats
            if 'spread' in data.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Spread", f"{data['spread'].iloc[-1]:.1f}")
                with col2:
                    st.metric("Spread Volatility", f"{data['spread'].std():.2f}")
            
            if 'total' in data.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Total", f"{data['total'].iloc[-1]:.1f}")
                with col2:
                    st.metric("Total Volatility", f"{data['total'].std():.2f}")
        
        # Market Context
        with st.expander("🌐 Market Context"):
            if st.session_state.current_recommendations:
                context = st.session_state.current_recommendations.get('market_context', {})
                st.write(f"**Volatility:** {context.get('volatility', 'Unknown')}")
                st.write(f"**Data Points:** {context.get('data_points', 0)}")
        
        # Betting Strategy
        with st.expander("🎯 Betting Strategy"):
            if st.session_state.current_recommendations:
                strategy = st.session_state.current_recommendations.get('betting_strategy', {})
                st.write(f"**Primary Approach:** {strategy.get('primary_approach', '')}")
                
                st.write("**Recommended Actions:**")
                for action in strategy.get('recommended_actions', []):
                    st.write(f"• {action}")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("📈 View Data Table"):
            self.show_data_table()
        
        if st.button("🔄 Update Analysis"):
            self.update_analysis()
        
        if st.button("💾 Export Results"):
            self.export_results()
    
    def show_data_table(self):
        """Display the historical data table"""
        if not st.session_state.historical_data.empty:
            st.subheader("📋 Historical Data")
            st.dataframe(st.session_state.historical_data, use_container_width=True)
    
    def update_analysis(self):
        """Update the analysis with current data"""
        if not st.session_state.historical_data.empty:
            with st.spinner("Updating analysis..."):
                signals = self.ta_engine.generate_trading_signals(
                    current_data=st.session_state.historical_data.iloc[-1].to_dict(),
                    historical_data=st.session_state.historical_data
                )
                
                recommendations = self.rec_engine.generate_comprehensive_recommendations(
                    signals, st.session_state.historical_data
                )
                
                st.session_state.current_recommendations = recommendations
                st.rerun()
    
    def export_results(self):
        """Export results to JSON"""
        if st.session_state.current_recommendations:
            import json
            from datetime import datetime
            
            filename = f"betting_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to serializable format
            export_data = json.loads(json.dumps(
                st.session_state.current_recommendations, 
                default=str, 
                indent=2
            ))
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(export_data, indent=2),
                file_name=filename,
                mime="application/json"
            )
    
    def clear_data(self):
        """Clear all data"""
        st.session_state.historical_data = pd.DataFrame()
        st.session_state.current_recommendations = {}
        st.session_state.analysis_history = []
        st.rerun()
    
    def save_analysis(self):
        """Save current analysis to history"""
        if st.session_state.current_recommendations:
            st.success("Analysis saved to history!")
    
    def view_analysis_history(self):
        """View analysis history"""
        if st.session_state.analysis_history:
            st.subheader("📚 Analysis History")
            
            for i, analysis in enumerate(st.session_state.analysis_history[::-1]):  # Show latest first
                with st.expander(f"Analysis {i+1} - {analysis['game']} ({analysis['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                    st.write(f"**Sport:** {analysis['sport']}")
                    st.write(f"**Top Pick:** {analysis['recommendations'].get('top_pick', {}).get('selection', 'N/A')}")
        
        else:
            st.info("No analysis history available.")

def main():
    """Main application entry point"""
    app = SportsBettingApp()
    app.main()

if __name__ == "__main__":
    main()
