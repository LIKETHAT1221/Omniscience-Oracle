import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Optional, Union

class OmniscienceDataParser:
    def __init__(self):
        self.parsed_data = []
        self._time_series_data = {}
        
    def parse_main_blocks(self, main_data_text: str) -> List[Dict]:
        lines = main_data_text.strip().split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
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
                    continue
                    
                results.append(parsed)
                self.parsed_data.append(parsed)
            except Exception as e:
                print(f"Error parsing block: {e}")
                continue
                
        return results
        
    def parse_splits_blocks(self, splits_data_text: str) -> List[Dict]:
        lines = splits_data_text.strip().split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                continue
                
            current_block.append(stripped)
            if len(current_block) == 8:
                blocks.append(current_block)
                current_block = []
        
        if current_block:
            blocks.append(current_block)
            
        results = []
        for block in blocks:
            try:
                if len(block) == 8:
                    parsed = self._parse_splits_block(block)
                    results.append(parsed)
            except Exception as e:
                print(f"Error parsing splits block: {e}")
                continue
                
        return results
        
    def merge_data(self, main_data: List[Dict], splits_data: List[Dict]) -> List[Dict]:
        for i, game_data in enumerate(main_data):
            if i < len(splits_data):
                game_data.update(splits_data[i])
        return main_data
        
    def _is_header_line(self, line: str) -> bool:
        header_indicators = ['time', 'spread', 'total', 'ml', 'runline', 'open', 'date']
        lower_line = line.lower()
        return any(indicator in lower_line for indicator in header_indicators) or '\t' in line and len(line.split('\t')) > 3
        
    def _parse_5_line_block(self, block: List[str]) -> Dict:
        line1_parts = block[0].split('\t') if '\t' in block[0] else block[0].split()
        
        if len(line1_parts) < 4:
            line1_parts = block[0].split()
            
        datetime_str = f"{line1_parts[0]} {line1_parts[1]}"
        
        spread_val = line1_parts[-1]
        team_parts = line1_parts[2:-1]
        favorite_team = " ".join(team_parts) if team_parts else "Unknown"
        
        spread_vig = block[1].strip()
        total_line = block[2].strip()
        total_vig = block[3].strip()
        
        ml_parts = block[4].split('\t') if '\t' in block[4] else block[4].split()
        if len(ml_parts) < 2:
            raise ValueError(f"Invalid moneyline format: {block[4]}")
            
        away_ml, home_ml = ml_parts[0], ml_parts[1]
        
        parsed = {
            'game_type': 'spread',
            'datetime': datetime_str,
            'favorite_team': favorite_team,
            'spread': spread_val,
            'spread_vig': spread_vig,
            'total': total_line,
            'total_vig': total_vig,
            'away_ml': away_ml,
            'home_ml': home_ml,
            'parsed_at': datetime.now().isoformat(),
            'away_bets_pct': None,
            'home_bets_pct': None,
            'away_money_pct': None,
            'home_money_pct': None
        }
        
        self._convert_to_implied_probabilities(parsed)
        
        return parsed
        
    def _parse_4_line_block(self, block: List[str]) -> Dict:
        line1_parts = block[0].split('\t') if '\t' in block[0] else block[0].split()
        
        if len(line1_parts) < 5:
            line1_parts = block[0].split()
            
        datetime_str = f"{line1_parts[0]} {line1_parts[1]}"
        away_ml = line1_parts[2]
        home_ml = line1_parts[3]
        total = line1_parts[4]
        
        total_vig = block[1].strip()
        
        runline_info = block[2].strip()
        runline_parts = runline_info.split('\t') if '\t' in runline_info else runline_info.split()
        if len(runline_parts) < 2:
            raise ValueError(f"Invalid runline format: {runline_info}")
            
        runline_team, runline = runline_parts[0], runline_parts[1]
        runline_vig = block[3].strip()
        
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
            'away_bets_pct': None,
            'home_bets_pct': None,
            'away_money_pct': None,
            'home_money_pct': None
        }
        
        self._convert_to_implied_probabilities(parsed)
        
        return parsed
        
    def _parse_splits_block(self, block: List[str]) -> Dict:
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
        for field in ['away_ml', 'home_ml']:
            if field in data and data[field]:
                data[f'{field}_implied'] = self._odds_to_implied_probability(data[field])
                
        for field in ['spread_vig', 'total_vig', 'runline_vig']:
            if field in data and data[field]:
                data[f'{field}_implied'] = self._odds_to_implied_probability(data[field])
                
    def _odds_to_implied_probability(self, odds_str: str) -> float:
        if not odds_str or str(odds_str).lower() == 'nan':
            return None
            
        if str(odds_str).lower() == 'even':
            return 0.5
            
        try:
            odds = float(odds_str)
        except ValueError:
            return None
            
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
            
    def add_to_time_series(self, game_data: Dict) -> str:
        if game_data['game_type'] == 'spread':
            game_key = f"{game_data['favorite_team']}_{game_data['datetime']}"
        else:
            game_key = f"{game_data['away_ml']}_{game_data['home_ml']}_{game_data['datetime']}"
            
        if game_key not in self._time_series_data:
            self._time_series_data[game_key] = []
            
        self._time_series_data[game_key].append(game_data)
        return game_key
        
    def get_time_series(self, game_key: str) -> List[Dict]:
        return self._time_series_data.get(game_key, [])
        
    def clear_data(self):
        self.parsed_data = []
        self._time_series_data = {}
