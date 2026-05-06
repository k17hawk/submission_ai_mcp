"""
Named Entity Recognition using spaCy.
Extracts locations (cities, states), dates, organizations from text.
"""

import spacy
from typing import List, Dict, Optional, Tuple,Any


class SpacyNERExtractor:
    """
    Uses spaCy's named entity recognition to extract:
    - GPE (cities, states, countries)
    - DATE (dates in various formats)
    - ORG (organizations — police departments, body shops)
    - PERSON (names — claimants, witnesses)
    """
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """Load spaCy model. Call once at server startup."""
        self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text: str, max_chars: int = 100000) -> Dict[str, List[Dict]]:
        """
        Extract all named entities from text.
        
        Returns:
            Dictionary grouped by entity type
        """
        doc = self.nlp(text[:max_chars])
        
        entities = {
            'GPE': [],      # Geopolitical entities (cities, states)
            'DATE': [],     # Dates
            'ORG': [],      # Organizations
            'PERSON': [],   # People
            'MONEY': [],    # Monetary amounts
            'CARDINAL': [], # Numbers
            'TIME': [],     # Times
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_,
                })
        
        return entities
    
    def extract_location(self, text: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Extract city and state from text.
        
        Returns:
            Tuple of (city, state_abbreviation, confidence)
        """
        entities = self.extract_entities(text)
        gpe_entities = entities.get('GPE', [])
        
        city = None
        state = None
        confidence = 0.0
        
        # States are usually 2-letter uppercase
        for ent in gpe_entities:
            if len(ent['text']) == 2 and ent['text'].isupper():
                state = ent['text']
                confidence = 0.70
                break
        
        # Cities are longer
        for ent in gpe_entities:
            if len(ent['text']) > 2 and ent['text'] != state:
                city = ent['text']
                if not city:
                    confidence = max(confidence, 0.60)
                break
        
        if city and state:
            confidence = 0.80
        
        return city, state, confidence
    
    def extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all dates from text with positions.
        """
        entities = self.extract_entities(text)
        return entities.get('DATE', [])
    
    def extract_organizations(self, text: str) -> List[str]:
        """
        Extract organization names (police departments, body shops, hospitals).
        """
        entities = self.extract_entities(text)
        return [e['text'] for e in entities.get('ORG', [])]
    
    def extract_people(self, text: str) -> List[str]:
        """
        Extract person names.
        """
        entities = self.extract_entities(text)
        return [e['text'] for e in entities.get('PERSON', [])]


# Quick test
if __name__ == "__main__":
    extractor = SpacyNERExtractor("en_core_web_lg")
    
    text = "The accident occurred on Highway 101 in Springfield, IL on May 12, 2023. Officer Johnson from Springfield PD filed report #88421."
    
    city, state, conf = extractor.extract_location(text)
    print(f"Location: {city}, {state} (confidence: {conf})")
    
    dates = extractor.extract_dates(text)
    print(f"Dates: {[d['text'] for d in dates]}")
    
    orgs = extractor.extract_organizations(text)
    print(f"Organizations: {orgs}")