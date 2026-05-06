"""
MCP Server 1: Submission Parser
Orchestrates extraction across regex, spaCy NER, and normalizer.
Exposes MCP tools: parse_submission, parse_multiple_documents, validate_extraction.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field

from src.mcp_submission_parsing.extractor.regex_extractor import RegexExtractor
from src.mcp_submission_parsing.extractor.spacy_ner import SpacyNERExtractor
from src.mcp_submission_parsing.extractor.field_normalizer import FieldNormalizer
from src.mcp_submission_parsing.config import get_patterns, get_normalizer_config
import re
# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedSubmission:
    policy_number: Optional[str] = None
    incident_date: Optional[str] = None
    incident_type: Optional[str] = None
    incident_severity: Optional[str] = None
    incident_location: Optional[str] = None
    incident_city: Optional[str] = None
    incident_state: Optional[str] = None
    auto_make: Optional[str] = None
    auto_model: Optional[str] = None
    auto_year: Optional[int] = None
    authorities_contacted: Optional[str] = None
    collision_type: Optional[str] = None
    number_of_vehicles_involved: Optional[int] = None
    bodily_injuries: Optional[int] = None
    witnesses: Optional[int] = None
    police_report_available: Optional[str] = None
    property_damage: Optional[str] = None
    total_claim_amount: Optional[float] = None
    incident_hour_of_the_day: Optional[int] = None
    
    extraction_confidence: float = 0.0
    missing_fields: List[str] = field(default_factory=list)
    source_document_type: Optional[str] = None
    
    def get_missing_required(self) -> List[str]:
        required = ['policy_number', 'incident_date', 'incident_type', 'auto_make']
        return [f for f in required if getattr(self, f) is None]
    
    def is_complete(self) -> bool:
        return len(self.get_missing_required()) == 0


class SubmissionParser:
    """Orchestrates the three extraction engines."""
    
    def __init__(self):
        patterns = get_patterns()
        normalizer_config = get_normalizer_config()
        
        self.regex_extractor = RegexExtractor(patterns)
        self.ner_extractor = SpacyNERExtractor("en_core_web_lg")
        self.normalizer = FieldNormalizer(normalizer_config)
    
    def parse(self, text: str, doc_type: str = "unknown") -> ParsedSubmission:
        """Main entry point: parse unstructured text into structured fields."""
        result = ParsedSubmission(source_document_type=doc_type)
        confidence_scores = []
        
        # ── Vehicle: Try combined year-make-model pattern FIRST ────────────
        # This handles "2020 Toyota Camry" in a single regex match
        groups, conf = self.regex_extractor.extract_all_groups('year_make_model', text)
        if groups and len(groups) >= 3:
            year_str, make_str, model_str = groups[0], groups[1], groups[2]
            year_val = self.normalizer.normalize_auto_year(year_str)
            make_val = self.normalizer.normalize_auto_make(make_str)
            if year_val:
                result.auto_year = year_val
                confidence_scores.append(conf)
            if make_val:
                result.auto_make = make_val
                confidence_scores.append(conf)
            if model_str:
                result.auto_model = model_str.strip()
                confidence_scores.append(0.78)
        
        # ── Vehicle: Fall back to individual patterns ──────────────────────
        # Try standalone auto_year if combined pattern didn't work
        if not result.auto_year:
            year_val, conf = self.regex_extractor.extract('auto_year', text)
            if year_val:
                year = self.normalizer.normalize_auto_year(year_val)
                if year:
                    result.auto_year = year
                    confidence_scores.append(conf)
        
        # Try make_model separately if no make yet
        if not result.auto_make:
            make_model_val, conf = self.regex_extractor.extract('make_model', text)
            if make_model_val:
                parts = make_model_val.split(' ', 1)
                if len(parts) >= 1:
                    result.auto_make = self.normalizer.normalize_auto_make(parts[0])
                    confidence_scores.append(conf)
                if len(parts) >= 2:
                    result.auto_model = parts[1]
        
        # ── All other regex-based extractions ──────────────────────────────
        regex_fields = [
            ('policy_number', 'policy_number', 'str'),
            ('incident_date', 'incident_date', 'date'),
            ('incident_type', 'incident_type', 'str'),
            ('incident_severity', 'incident_severity', 'str'),
            ('incident_location', 'incident_location', 'str'),
            ('authorities_contacted', 'authorities_contacted', 'str'),
            ('collision_type', 'collision_type', 'str'),
            ('number_of_vehicles', 'number_of_vehicles_involved', 'int'),
            ('bodily_injuries', 'bodily_injuries', 'int'),
            ('witnesses', 'witnesses', 'int'),
            ('police_report', 'police_report_available', 'yesno'),
            ('property_damage', 'property_damage', 'yesno'),
            ('total_claim_amount', 'total_claim_amount', 'float'),
        ]
        
        for pattern_key, field_name, field_type in regex_fields:
            value, conf = self.regex_extractor.extract(pattern_key, text)
            if value and conf > 0.5:
                normalized = self._normalize_value(field_name, value, field_type)
                if normalized is not None:
                    setattr(result, field_name, normalized)
                    confidence_scores.append(conf)
        
        # ── spaCy NER: city and state ──────────────────────────────────────
        city, state, conf = self.ner_extractor.extract_location(text)
        if city:
            result.incident_city = city
            confidence_scores.append(conf)
        if state:
            result.incident_state = state
            confidence_scores.append(conf)
        
        # ── Normalize date if extracted in non-standard format ─────────────
        if result.incident_date:
            normalized_date = self.normalizer.normalize_date(result.incident_date)
            if normalized_date:
                result.incident_date = normalized_date
        
        # ── Finalize ───────────────────────────────────────────────────────
        if confidence_scores:
            result.extraction_confidence = round(
                sum(confidence_scores) / len(confidence_scores), 4
            )
        
        result.missing_fields = result.get_missing_required()
        return result
    
    def _normalize_value(self, field_name: str, value: str, field_type: str) -> Any:
        """Normalize raw extracted value to proper type."""
        if field_type == 'str':
            return self.normalizer.normalize(field_name, value) or value.strip()
        elif field_type == 'int':
            try:
                return int(value.strip())
            except ValueError:
                return None
        elif field_type == 'float':
            return self.normalizer.normalize_amount(value)
        elif field_type == 'date':
            return self.normalizer.normalize_date(value)
        elif field_type == 'yesno':
            return self.normalizer.normalize_yes_no(value)
        return value
    
    def parse_multiple(self, documents: List[Dict[str, str]]) -> ParsedSubmission:
        """Parse multiple documents and merge results."""
        all_results = []
        for doc in documents:
            result = self.parse(doc['text'], doc.get('doc_type', 'unknown'))
            all_results.append(result)
        
        merged = ParsedSubmission(
            source_document_type=", ".join(d.get('doc_type', 'unknown') for d in documents)
        )
        
        field_names = [
            f.name for f in ParsedSubmission.__dataclass_fields__.values()
            if f.name not in ('extraction_confidence', 'missing_fields', 'source_document_type')
        ]
        
        for field in field_names:
            for r in all_results:
                val = getattr(r, field)
                if val is not None:
                    setattr(merged, field, val)
                    break
        
        merged.extraction_confidence = max(
            (r.extraction_confidence for r in all_results), default=0.0
        )
        merged.missing_fields = merged.get_missing_required()
        return merged


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON + MCP TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

parser = SubmissionParser()

def parse_submission(text: str, doc_type: str = "unknown") -> Dict[str, Any]:
    return asdict(parser.parse(text, doc_type))

def parse_multiple_documents(documents: List[Dict[str, str]]) -> Dict[str, Any]:
    return asdict(parser.parse_multiple(documents))

def validate_extraction(parsed: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    if parsed.get('policy_number') and not re.match(r'^POL-\d{6}$', str(parsed['policy_number'])):
        errors.append(f"Invalid policy_number format: {parsed['policy_number']}")
    if parsed.get('auto_year') and not (1980 <= int(parsed['auto_year']) <= 2025):
        errors.append(f"auto_year out of range: {parsed['auto_year']}")
    if parsed.get('total_claim_amount') and float(parsed['total_claim_amount']) <= 0:
        errors.append("total_claim_amount must be positive")
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': [],
        'completeness': 1.0 - (len(parsed.get('missing_fields', [])) / 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    texts = [
        "I drive a 2020 Toyota Camry. Policy is POL-111222.",
        "a 2019 Honda Civic was damaged. Policy POL-333444.",
        "My 2022 Ford F-150 was hit. POL-555666.",
        "Policy POL-123456. I have a 2018 Chevrolet Malibu. Accident on 2023-01-15.",
    ]
    
    for t in texts:
        r = parser.parse(t, "email")
        print(f"Text: {t}")
        print(f"  Year={r.auto_year}, Make={r.auto_make}, Model={r.auto_model}")
        print(f"  Policy={r.policy_number}, Confidence={r.extraction_confidence}")
        print()