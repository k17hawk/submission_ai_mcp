"""
Pattern-based extraction using compiled regex.
Handles structured patterns like policy numbers, dates, amounts.
"""

import re
from typing import Optional, Dict, Any, List, Tuple


class RegexExtractor:
    """
    Extracts fields using pre-compiled regex patterns.
    All patterns are loaded from field_definitions.yaml at init.
    """
    
    def __init__(self, patterns_config: Dict[str, Any]):
        self.patterns = {}
        self._compile_all(patterns_config)
    
    def _compile_all(self, config: Dict[str, Any]) -> None:
        """Compile all patterns from config dictionary"""
        for field_name, pattern_list in config.items():
            if isinstance(pattern_list, str):
                self.patterns[field_name] = re.compile(pattern_list, re.IGNORECASE)
            elif isinstance(pattern_list, list):
                self.patterns[field_name] = [
                    re.compile(p, re.IGNORECASE) for p in pattern_list
                ]
    
    def extract(self, field_name: str, text: str) -> Tuple[Optional[str], float]:
        """
        Extract a field from text using its compiled pattern.
        
        Returns:
            Tuple of (extracted_value, confidence_score)
        """
        if field_name not in self.patterns:
            return None, 0.0
        
        pattern = self.patterns[field_name]
        
        # Single pattern
        if isinstance(pattern, re.Pattern):
            match = pattern.search(text)
            if match:
                return match.group(1).strip() if match.lastindex else match.group(0).strip(), 0.90
            return None, 0.0
        
        # Multiple patterns — try in order, first match wins
        for idx, pat in enumerate(pattern):
            match = pat.search(text)
            if match:
                confidence = 0.95 - (idx * 0.05)
                return match.group(1).strip() if match.lastindex else match.group(0).strip(), max(confidence, 0.70)
        
        return None, 0.0
    
    def extract_all_groups(self, field_name: str, text: str) -> Tuple[Optional[tuple], float]:
        """
        Extract ALL capture groups from a pattern match.
        Use this for patterns with multiple groups (e.g., year + make + model).
        
        Returns:
            Tuple of (groups_tuple, confidence_score)
            groups_tuple is None if no match, otherwise contains all group values
        """
        if field_name not in self.patterns:
            return None, 0.0
        
        pattern = self.patterns[field_name]
        
        # Get the first pattern (single or from list)
        if isinstance(pattern, list):
            pattern = pattern[0]
        
        match = pattern.search(text)
        if match and match.lastindex and match.lastindex >= 2:
            groups = match.groups()
            return groups, 0.92
        
        return None, 0.0
    
    def extract_field_with_context(self, field_name: str, text: str) -> Dict[str, Any]:
        """
        Extract a field and return full context including the match span.
        """
        if field_name not in self.patterns:
            return {'value': None, 'confidence': 0.0, 'match_text': None, 'span': None}
        
        pattern = self.patterns[field_name]
        patterns_to_try = pattern if isinstance(pattern, list) else [pattern]
        
        for idx, pat in enumerate(patterns_to_try):
            match = pat.search(text)
            if match:
                confidence = 0.95 - (idx * 0.05)
                return {
                    'value': match.group(1).strip() if match.lastindex else match.group(0).strip(),
                    'confidence': max(confidence, 0.70),
                    'match_text': match.group(0),
                    'span': match.span(),
                    'all_groups': match.groups() if match.lastindex else None,
                }
        
        return {'value': None, 'confidence': 0.0, 'match_text': None, 'span': None, 'all_groups': None}