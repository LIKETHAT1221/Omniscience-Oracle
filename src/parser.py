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
    
    # ... [rest of your existing parser methods] ...
