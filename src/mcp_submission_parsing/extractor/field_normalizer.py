"""
Maps extracted text to valid categorical values.
Handles fuzzy matching, synonyms, and misspellings.
"""

from typing import Optional, Dict, List, Any
import re


class FieldNormalizer:
    """
    Converts raw extracted text into standardized categorical values.
    Uses reference tables for valid values and synonym mappings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.mappings = config.get('mappings', {})
        self.valid_values = config.get('valid_values', {})
        self.auto_makes = config.get('auto_makes', [
            "Toyota", "Honda", "Ford", "Chevrolet", "BMW",
            "Nissan", "Jeep", "Dodge", "Hyundai", "Kia"
        ])
    
    def normalize(self, field_name: str, raw_value: str) -> Optional[str]:
        """
        Normalize a raw value to its canonical form.
        
        Args:
            field_name: The field being normalized (e.g., 'incident_type')
            raw_value: The raw extracted text
        
        Returns:
            Canonical value or None if no match found
        """
        if not raw_value:
            return None
        
        raw_lower = raw_value.lower().strip()
        
        # Check if already valid
        valid_vals = self.valid_values.get(field_name, [])
        for v in valid_vals:
            if raw_lower == v.lower():
                return v
        
        # Check synonym mappings
        field_mappings = self.mappings.get(field_name, {})
        for key, canonical in field_mappings.items():
            if key in raw_lower or raw_lower in key:
                return canonical
        
        return None
    
    def normalize_auto_make(self, raw_make: str) -> Optional[str]:
        """
        Special handling for auto makes with common misspellings.
        """
        if not raw_make:
            return None
        
        raw_lower = raw_make.lower().strip()
        
        # Handle common misspellings
        make_fixes = {
            'toyota': 'Toyota',
            'toy': 'Toyota',
            'honda': 'Honda',
            'hon': 'Honda',
            'ford': 'Ford',
            'chevy': 'Chevrolet',
            'chev': 'Chevrolet',
            'cheverolet': 'Chevrolet',
            'bmw': 'BMW',
            'nissan': 'Nissan',
            'nis': 'Nissan',
            'jeep': 'Jeep',
            'dodge': 'Dodge',
            'ram': 'Dodge',
            'hyundai': 'Hyundai',
            'hyun': 'Hyundai',
            'kia': 'Kia',
        }
        
        for key, canonical in make_fixes.items():
            if key in raw_lower:
                return canonical
        
        # Check against valid makes
        for make in self.auto_makes:
            if make.lower() in raw_lower or raw_lower in make.lower():
                return make
        
        return raw_make.strip().title()
    
    def normalize_auto_year(self, raw_year: str) -> Optional[int]:
        """
        Validate and normalize auto year.
        """
        if not raw_year:
            return None
        
        try:
            year = int(raw_year.strip())
            if 1980 <= year <= 2025:
                return year
        except ValueError:
            pass
        
        # Try extracting year from longer string
        match = re.search(r'\b(19[89]\d|20[0-2]\d)\b', str(raw_year))
        if match:
            year = int(match.group(1))
            if 1980 <= year <= 2025:
                return year
        
        return None
    
    def normalize_date(self, raw_date: str) -> Optional[str]:
        """
        Normalize various date formats to YYYY-MM-DD.
        """
        if not raw_date:
            return None
        
        from datetime import datetime
        
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y',
            '%m/%d/%y',
        ]
        
        raw_clean = raw_date.strip()
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(raw_clean, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def normalize_amount(self, raw_amount: str) -> Optional[float]:
        """
        Normalize monetary amounts.
        """
        if not raw_amount:
            return None
        
        cleaned = raw_amount.strip()
        cleaned = cleaned.replace('$', '').replace(',', '')
        
        try:
            return float(cleaned)
        except ValueError:
            pass
        
        return None
    
    def normalize_yes_no(self, raw_value: str) -> Optional[str]:
        """
        Normalize YES/NO/? values.
        """
        if not raw_value:
            return None
        
        raw_upper = raw_value.strip().upper()
        
        if raw_upper in ['YES', 'Y', 'TRUE', 'AVAILABLE', 'FILED']:
            return 'YES'
        elif raw_upper in ['NO', 'N', 'FALSE', 'UNAVAILABLE', 'NOT FILED']:
            return 'NO'
        elif raw_upper in ['?', 'UNKNOWN', 'N/A', 'NA']:
            return '?'
        
        return None


# Quick test
if __name__ == "__main__":
    config = {
        'mappings': {
            'incident_type': {
                'single vehicle': 'Single Vehicle Collision',
                'multi vehicle': 'Multi-vehicle Collision',
                'parked': 'Parked Car',
                'theft': 'Vehicle Theft',
                'stolen': 'Vehicle Theft',
            },
            'incident_severity': {
                'minor': 'Minor Damage',
                'major': 'Major Damage',
                'severe': 'Major Damage',
                'total loss': 'Total Loss',
                'totaled': 'Total Loss',
                'trivial': 'Trivial Damage',
                'small': 'Trivial Damage',
            },
        },
        'valid_values': {
            'incident_type': [
                'Single Vehicle Collision', 'Multi-vehicle Collision',
                'Parked Car', 'Vehicle Theft', 'Other'
            ],
            'incident_severity': [
                'Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'
            ],
        }
    }
    
    normalizer = FieldNormalizer(config)
    
    print(normalizer.normalize('incident_type', 'single vehicle crash'))
    print(normalizer.normalize('incident_severity', 'totaled'))
    print(normalizer.normalize_auto_make('chevy'))
    print(normalizer.normalize_date('05/12/2023'))
    print(normalizer.normalize_amount('$4,520.00'))
    print(normalizer.normalize_yes_no('Y'))