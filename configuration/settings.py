# Configuration and toggle switches for the Omniscience system

class Config:
    def __init__(self):
        # Parser toggles
        self.parse_4_line_blocks = True
        self.parse_5_line_blocks = True
        self.parse_splits_data = True
        self.ignore_header_rows = True
        
        # TA Indicator toggles
        self.use_momentum_indicators = True
        self.use_rsi = True
        self.use_fibonacci = True
        self.use_z_score = True
        self.use_steam_detection = True
        self.use_adaptive_ma = True
        self.use_greeks = True
        
        # Threshold settings
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.z_score_threshold = 2.0
        self.steam_confidence_threshold = 0.6
        
        # Recommendation settings
        self.min_confidence_for_action = 0.6
        self.strong_confidence_threshold = 0.75
        self.kelly_fraction = 0.25  # Fraction of full Kelly to use
        
    def toggle_parser_mode(self, mode: str):
        """Toggle between different parsing modes"""
        if mode == '4_line_only':
            self.parse_4_line_blocks = True
            self.parse_5_line_blocks = False
        elif mode == '5_line_only':
            self.parse_4_line_blocks = False
            self.parse_5_line_blocks = True
        elif mode == 'both_modes':
            self.parse_4_line_blocks = True
            self.parse_5_line_blocks = True
        elif mode == 'no_splits':
            self.parse_splits_data = False
            
    def toggle_indicator(self, indicator: str, enabled: bool):
        """Toggle specific indicators on/off"""
        if hasattr(self, f'use_{indicator}'):
            setattr(self, f'use_{indicator}', enabled)
            
    def get_active_indicators(self) -> List[str]:
        """Get list of active indicators"""
        return [indicator for indicator in [
            'momentum_indicators', 'rsi', 'fibonacci', 'z_score',
            'steam_detection', 'adaptive_ma', 'greeks'
        ] if getattr(self, f'use_{indicator}', False)]

# Global configuration instance
config = Config()
