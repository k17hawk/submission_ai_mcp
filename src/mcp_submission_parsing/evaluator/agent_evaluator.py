"""
Rule‑based evaluator for extracted claim fields.
No LLM – uses regex and simple logic to verify extracted values.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


TARGET_FIELDS = [
    "policy_number", "customer_id", "customer_name", "incident_date",
    "incident_type", "collision_type", "incident_location", "incident_state",
    "auto_year", "auto_make", "auto_model", "number_of_vehicles",
    "witnesses", "total_claim_amount", "police_report_available",
    "incident_hour_of_the_day"
]

# Patterns for verifying each field from source text
VERIFICATION_PATTERNS = {
    # Must match full POL-XXXXXX token
    "policy_number": r'\b(POL-\d{6})\b',
    # Must match full CUST-XXXXX token
    "customer_id": r'\b(CUST-\d{5,})\b',
    # Stop before 'with', 'CUST', 'policy' — use lookahead not greedy capture
    "customer_name": r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+?)(?=\s+with\b|\s+CUST\b|\s+policy\b|$)',
    # ISO date only
    "incident_date": r'\b(\d{4}-\d{2}-\d{2})\b',
    "incident_type": {
        "Multi-vehicle Collision": r'\d+[-\s]?(?:car|vehicle)s?\s+(?:pileup|collision|accident)',
        "Single Vehicle Collision": r'single\s+(?:car|vehicle)',
        "Parked Car": r'parked',
        "Vehicle Theft": r'(?:theft|stolen)',
        "Other": r'.*'
    },
    "collision_type": {
        "Front Collision": r'\bFront\s+(?:Collision|End|Impact)\b',
        "Rear Collision": r'\bRear\s+(?:Collision|End|Impact)\b',
        "Side Collision": r'\bSide\s+(?:Collision|Impact)\b|T-?bone',
    },
    "incident_location": r'(I-\d+|Highway\s*\d+|Parking\s+Lot|Local\s+Road|Interstate)',
    # Must be a known US state abbreviation, not any 2-letter word like "in"
    "incident_state": r'\b(A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AT]|W[AVIY])\b',
    "auto_year": r'\b(19[8-9]\d|20[0-2]\d)\b',
    "auto_make": r'\b(Toyota|Honda|Ford|Chevrolet|BMW|Nissan|Jeep|Dodge|Hyundai|Kia|Subaru|Tesla|Mercedes|Audi)\b',
    # Must be preceded by a known make to avoid matching sentence fragments
    "auto_model": r'(?:Toyota|Honda|Ford|Chevrolet|BMW|Nissan|Jeep|Dodge|Hyundai|Kia|Subaru|Tesla|Mercedes|Audi)\s+([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+)?)',
    "number_of_vehicles": r'(\d+)[-\s]?(?:car|vehicle)s?\s+(?:pileup|collision|involved)',
    "witnesses": r'(\d+)\s+witness',
    # Must start with $ or follow damage/claim keyword — avoid matching bare digits
    "total_claim_amount": r'\$\s*([\d,]+(?:\.\d{2})?)',
    # Verify as YES/NO based on text evidence, not raw string
    "police_report_available": r'police\s+report\s+(?:available|filed|submitted)',
}

class RuleEvaluator:
    """Evaluates extracted fields against source text using deterministic rules."""

    def __init__(self):
        self.verification = VERIFICATION_PATTERNS

    def evaluate(self, extracted: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        """
        Returns:
            {
                "field_evaluations": { field: {"status": "ok|issue|missing", "reason": str, "corrected_value": Any} },
                "errors": List[Dict],
                "corrected_extraction": Dict
            }
        """
        evaluations = {}
        errors = []
        corrected = extracted.copy()

        for field in TARGET_FIELDS:
            extracted_val = extracted.get(field)
            status = "ok"
            reason = ""
            corrected_val = None

            # 1. If field is missing from extraction
            if extracted_val is None or extracted_val == "":
                # Check if value can be found in source text
                found, val = self._find_in_text(field, source_text)
                if found:
                    status = "missing"
                    reason = f"Field should be '{val}' but was not extracted."
                    corrected_val = val
                    errors.append({"field": field, "extracted": None, "corrected": val, "reason": reason})
                else:
                    status = "ok"  # genuinely absent
                    reason = "Information not present in source text."
                evaluations[field] = {"status": status, "reason": reason, "corrected_value": corrected_val}
                if corrected_val is not None:
                    corrected[field] = corrected_val
                continue

            # 2. Field exists – verify correctness
            is_correct, found_in_text, suggested = self._verify_field(field, extracted_val, source_text)
            if is_correct:
                status = "ok"
                reason = f"'{extracted_val}' matches source text."
            else:
                status = "issue"
                if found_in_text and suggested is not None:
                    reason = f"Extracted '{extracted_val}' but text indicates '{suggested}'."
                    corrected_val = suggested
                    errors.append({"field": field, "extracted": extracted_val, "corrected": suggested, "reason": reason})
                else:
                    reason = f"Extracted '{extracted_val}' but could not confirm in text (maybe absent or ambiguous)."

            evaluations[field] = {"status": status, "reason": reason, "corrected_value": corrected_val}
            if corrected_val is not None:
                corrected[field] = corrected_val

        # Build final report
        ok_count = sum(1 for e in evaluations.values() if e["status"] == "ok")
        issue_count = sum(1 for e in evaluations.values() if e["status"] == "issue")
        missing_count = sum(1 for e in evaluations.values() if e["status"] == "missing")

        return {
            "field_evaluations": evaluations,
            "errors": errors,
            "corrected_extraction": corrected,
            "summary": {
                "total": len(TARGET_FIELDS),
                "ok": ok_count,
                "issues": issue_count,
                "missing": missing_count,
                "passed": issue_count == 0 and missing_count == 0
            }
        }

    def _find_in_text(self, field: str, text: str) -> Tuple[bool, Any]:
        """Try to extract correct value from source text for a missing field."""
        pattern = self.verification.get(field)
        if not pattern:
            return False, None
        if isinstance(pattern, dict):
            return False, None
        flags = 0 if field == "incident_state" else re.IGNORECASE
        match = re.search(pattern, text, flags)
        if match:
            val = match.group(1).strip() if match.lastindex else match.group(0).strip()
            return True, val
        return False, None

    def _verify_field(self, field: str, extracted_val: Any, text: str) -> Tuple[bool, bool, Any]:
        """
        Returns:
            (is_correct, found_in_text, suggested_correction)
        """
        pattern = self.verification.get(field)
        if not pattern:
            # No verification rule – assume correct
            return True, True, None

        # Special handling for derived field
        if field == "incident_hour_of_the_day":
            # Check if it's a valid hour (0-23) and consistent with "X hours ago"
            hours_ago_match = re.search(r'(\d+)\s*hours?\s+ago', text, re.IGNORECASE)
            if hours_ago_match:
                # We would need submission time to compute expected hour – skip for now
                return True, True, None
            return 0 <= int(extracted_val) <= 23, True, None

        # Special handling for police_report_available — pattern proves presence, not value
        if field == "police_report_available":
            report_in_text = bool(re.search(
                r'police\s+report\s+(?:available|filed|submitted)', text, re.IGNORECASE
            ))
            no_report_in_text = bool(re.search(r'no\s+police\s+report', text, re.IGNORECASE))
            if report_in_text:
                return extracted_val == "YES", True, "YES"
            elif no_report_in_text:
                return extracted_val == "NO", True, "NO"
            return True, False, None  # can't confirm either way

        # For categorical fields with multiple possible values
        if isinstance(pattern, dict):
            for canonical, pat in pattern.items():
                if re.search(pat, text, re.IGNORECASE):
                    if canonical == extracted_val:
                        return True, True, None
                    else:
                        return False, True, canonical
            return False, False, None

        # State codes must match case-sensitively (avoid "in" matching IN/Indiana)
        flags = 0 if field == "incident_state" else re.IGNORECASE

        # Simple regex pattern
        match = re.search(pattern, text, flags)
        if not match:
            return False, False, None

        matched = match.group(1) if match.lastindex else match.group(0)
        # Normalize matched string
        matched = matched.strip()

        # Compare with extracted value (allowing minor differences like extra spaces)
        if str(extracted_val).strip().lower() == matched.lower():
            return True, True, None
        else:
            return False, True, matched