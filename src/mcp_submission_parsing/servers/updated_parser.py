"""Agent 1: Submission Parser - Config-driven using regex extractor and field normalizer"""

import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_submission_parsing.extractor.field_normalizer import FieldNormalizer
from src.mcp_submission_parsing.extractor.regex_extractor import RegexExtractor
from src.mcp_submission_parsing.config import load_config, get_patterns, get_normalizer_config


class ParserAgent:
    """
    Parses claim submissions using:
    - RegexExtractor (patterns from field_definitions.yaml)
    - FieldNormalizer (synonyms, valid values, date/amount cleaning)
    - Optional LLM enhancement for missing critical fields
    """

    def __init__(self):
        # Load full configuration
        self.config = load_config()
        patterns = get_patterns()
        normalizer_config = get_normalizer_config()

        # constructor
        self.regex_extractor = RegexExtractor(patterns)
        self.field_normalizer = FieldNormalizer(normalizer_config)
        self.internal_only_fields = {
            'police_report',       
            'year_make_model',     
            'make_model',          
            'city_state',
        }

        # for post processing  fields
        self.derived_fields = {
            'incident_hour_of_the_day': self._derive_incident_hour,
            'police_report_available': self._normalize_police_report,
            'total_claim_amount': self._normalize_amount_field,
            'number_of_vehicles': self._normalize_int_fields,
            'bodily_injuries': self._normalize_int_fields,
            'witnesses': self._normalize_int_fields,
            'auto_year': self._normalize_auto_year,
            'auto_make': self._normalize_auto_make,
            'incident_date': self._normalize_date,
        }

        #tracking
        self.pattern_fields = list(patterns.keys())

    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process claim text and extract fields - config-driven"""

        text = payload.get('text', '')
        use_llm = payload.get('use_llm', False)

        if not text:
            return {
                'error': 'No claim text provided',
                'missing_fields': ['claim_text'],
                'extraction_confidence': 0.0
            }

        #regex extraction
        extracted = self._extract_all_fields(text)

        # normalization and derived fields
        extracted = self._post_process_fields(extracted, text)

        # determine missing critical fields based on config
        required_fields = self.config.get('required_fields', [
            'policy_number', 'incident_date', 'incident_type', 'auto_make'
        ])
        missing = [f for f in required_fields if not extracted.get(f)]

        # confidence scoring
        scoreable_fields = [f for f in self.pattern_fields if f not in self.internal_only_fields]
        total_possible = len(scoreable_fields)
        extracted_count = sum(1 for f in scoreable_fields if extracted.get(f))
        confidence = extracted_count / total_possible if total_possible > 0 else 0.0

        result = {
            **extracted,
            'extraction_confidence': round(confidence, 2),
            'missing_fields': missing,
            'llm_enhanced': False,
            'extraction_success': len(missing) == 0,
            'extracted_fields_count': extracted_count,
            'total_possible_fields': total_possible,
        }

        #LLM enhancement for missing critical fields
        if use_llm and missing:
            llm_fields = await self._extract_with_llm(text, missing)
            for key, value in llm_fields.items():
                if value and not extracted.get(key):
                    extracted[key] = value
                    result[key] = value
                    if key in missing:
                        missing.remove(key)

            result['llm_enhanced'] = True
            result['missing_fields'] = missing
            result['extraction_confidence'] = min(0.95, confidence + 0.2)

        return result

    # -------------------------------------------------------------------------
    # Extraction
    # -------------------------------------------------------------------------

    def _extract_all_fields(self, text: str) -> Dict[str, Any]:
        """Extract all fields defined in patterns using RegexExtractor"""
        # Always derived via custom logic — never taken raw from regex
        always_derived = {'incident_hour_of_the_day'}
        result = {}
        for field_name in self.pattern_fields:
            if field_name in always_derived:
                continue
            value, confidence = self.regex_extractor.extract(field_name, text)
            if value is not None:
                result[field_name] = value
        return result

    # -------------------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------------------

    def _post_process_fields(self, extracted: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Apply field-specific normalisation and derived field logic.
        Uses FieldNormalizer where possible, plus custom handlers.
        """
        processed = extracted.copy()

        # ------------------------------------------------------------------
        # 0. Pileup / multi-vehicle detection from raw text.
        #    The pileup regex pattern has no capture group (intentionally),
        #    so the extractor won't populate incident_type for "2-car pileup".
        #    We check raw text and set the canonical value directly.
        # ------------------------------------------------------------------
        if not processed.get('incident_type') or processed.get('incident_type', '').isdigit():
            if re.search(
                r'\d+[-\s]?(?:car|vehicle)s?\s+(?:pileup|collision|accident|crash)',
                text, re.IGNORECASE
            ):
                processed['incident_type'] = 'Multi-vehicle Collision'

        # ------------------------------------------------------------------
        # 1. Standard normalisation using FieldNormalizer
        # ------------------------------------------------------------------
        for field, value in list(processed.items()):
            if field in self.internal_only_fields:
                continue  # handled separately below

            normalized = None
            if field == 'incident_date':
                normalized = self.field_normalizer.normalize_date(value)
            elif field in ('total_claim_amount', 'property_damage_amount'):
                normalized = self.field_normalizer.normalize_amount(value)
            elif field in ('police_report_available', 'property_damage'):
                normalized = self.field_normalizer.normalize_yes_no(value)
            elif field == 'auto_make':
                normalized = self.field_normalizer.normalize_auto_make(value)
            elif field in ('incident_type', 'incident_severity', 'incident_location',
                           'authorities_contacted', 'collision_type'):
                normalized = self.field_normalizer.normalize(field, value)

            if normalized is not None:
                processed[field] = normalized

        # ------------------------------------------------------------------
        # 2. Special / derived fields
        # ------------------------------------------------------------------
        for derived_field, handler in self.derived_fields.items():
            # Skip if already present and non-empty
            if derived_field in processed and processed[derived_field] is not None:
                continue
            handler(processed, text)

        # ------------------------------------------------------------------
        # 3. Policy number prefix cleanup (ensure POL-XXXXXX)
        # ------------------------------------------------------------------
        if 'policy_number' in processed:
            pn = processed['policy_number']
            pn = re.sub(r'^POL[-\s]?', '', pn, flags=re.IGNORECASE)
            processed['policy_number'] = f"POL-{pn}"

        # ------------------------------------------------------------------
        # 4. Customer ID prefix cleanup — handled in step 7f below
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # 5. Customer name: strip trailing context words
        # ------------------------------------------------------------------
        if 'customer_name' in processed:
            name = processed['customer_name']
            name = re.sub(r'\s+(?:with|CUST[-\w]*|policy[-\w]*).*$', '', name, flags=re.IGNORECASE)
            processed['customer_name'] = name.strip().rstrip(',;:')

        # ------------------------------------------------------------------
        # 6. incident_state: must be exactly a 2-letter uppercase state code.
        # ------------------------------------------------------------------
        state = processed.get('incident_state', '')
        valid_states = set(self.config.get('valid_values', {}).get('incident_state', []))
        if not re.fullmatch(r'[A-Z]{2}', str(state)) or (valid_states and state not in valid_states):
            state_match = re.search(
                r'\b(A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|'
                r'N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AT]|W[AVIY])\b', text
            )
            if state_match:
                processed['incident_state'] = state_match.group(1)
            else:
                processed.pop('incident_state', None)

        # ------------------------------------------------------------------
        # 7. incident_location: prefer specific code like "I-95" over generic.
        # ------------------------------------------------------------------
        raw_location, _ = self.regex_extractor.extract('incident_location', text)
        if raw_location and re.match(r'^I-\d+$', raw_location.strip()):
            processed['incident_location'] = raw_location.strip()

        # ------------------------------------------------------------------
        # 7b. incident_city: strip trailing conjunctions/noise.
        # ------------------------------------------------------------------
        if 'incident_city' in processed:
            city = processed['incident_city']
            city = re.sub(r'\s+(?:and|the|or|at|in|near)\b.*$', '', city, flags=re.IGNORECASE).strip().rstrip(',;:')
            processed['incident_city'] = city if len(city) >= 2 else processed.pop('incident_city', None)

        # ------------------------------------------------------------------
        # 7c. auto_model: drop noise values that aren't real model names.
        # ------------------------------------------------------------------
        if 'auto_model' in processed:
            model = processed['auto_model']
            noise = {'was involved', 'involved', 'the vehicle', 'a vehicle'}
            if model.lower() in noise or len(model.split()) > 3 or not re.match(r'^[A-Z]', model):
                processed.pop('auto_model', None)

        # ------------------------------------------------------------------
        # 7d. total_claim_amount: ensure it's a float, not a raw string.
        # ------------------------------------------------------------------
        if 'total_claim_amount' in processed:
            val = processed['total_claim_amount']
            if not isinstance(val, float):
                normalized = self.field_normalizer.normalize_amount(str(val))
                if normalized is not None:
                    processed['total_claim_amount'] = normalized
                else:
                    processed.pop('total_claim_amount', None)

        # ------------------------------------------------------------------
        # 7e. police_report_available: normalize raw match to YES / NO / ?
        # ------------------------------------------------------------------
        if 'police_report_available' in processed:
            val = processed['police_report_available']
            if val not in ('YES', 'NO', '?'):
                normalized = self.field_normalizer.normalize_yes_no(str(val))
                if normalized:
                    processed['police_report_available'] = normalized
                elif re.search(r'police\s+report\s+(?:available|filed|submitted)', str(val), re.IGNORECASE):
                    processed['police_report_available'] = 'YES'
                elif re.search(r'no\s+police\s+report', str(val), re.IGNORECASE):
                    processed['police_report_available'] = 'NO'

        # ------------------------------------------------------------------
        # 7f. customer_id: ensure CUST-XXXXX format (pattern captures digits only).
        # ------------------------------------------------------------------
        if 'customer_id' in processed:
            cid = processed['customer_id']
            if not cid.upper().startswith('CUST'):
                processed['customer_id'] = f"CUST-{cid}"
            else:
                cid = re.sub(r'^CUST[-\s]?', '', cid, flags=re.IGNORECASE)
                processed['customer_id'] = f"CUST-{cid}"

        # ------------------------------------------------------------------
        # 8. Integer fields: ensure witnesses, bodily_injuries,
        #    number_of_vehicles are stored as int, not string.
        # ------------------------------------------------------------------
        for int_field in ('witnesses', 'bodily_injuries', 'number_of_vehicles'):
            if int_field in processed:
                try:
                    processed[int_field] = int(str(processed[int_field]).replace(',', ''))
                except (ValueError, TypeError):
                    processed.pop(int_field, None)

        # ------------------------------------------------------------------
        # 9. Remove internal-only pattern keys from the final output.
        #    These were used only as intermediate extraction helpers.
        # ------------------------------------------------------------------
        for field in self.internal_only_fields:
            processed.pop(field, None)

        return processed

    # -------------------------------------------------------------------------
    # Derived field handlers
    # -------------------------------------------------------------------------

    def _derive_incident_hour(self, extracted: Dict, text: str):
        """
        Derive incident_hour_of_the_day from:
        1. "X hours ago" phrasing  → current_hour - X (mod 24)
        2. Explicit time like "2:30 PM" in incident_time
        3. Direct hour pattern
        """
        # Try incident_time pattern first
        time_val, _ = self.regex_extractor.extract('incident_time', text)
        if time_val:
            hours_ago = re.search(r'(\d+)\s*hours?\s+ago', time_val, re.IGNORECASE)
            if hours_ago:
                current_hour = datetime.now().hour
                hour = (current_hour - int(hours_ago.group(1))) % 24
                extracted['incident_hour_of_the_day'] = hour
                return

            time_match = re.search(r'(\d{1,2})(?::\d{2})?\s*(AM|PM)', time_val, re.IGNORECASE)
            if time_match:
                hour = int(time_match.group(1))
                meridiem = time_match.group(2).upper()
                if meridiem == 'PM' and hour != 12:
                    hour += 12
                elif meridiem == 'AM' and hour == 12:
                    hour = 0
                extracted['incident_hour_of_the_day'] = hour
                return

        # Also check raw text for "X hours ago" in case incident_time pattern didn't fire
        raw_hours_ago = re.search(r'(\d+)\s*hours?\s+ago', text, re.IGNORECASE)
        if raw_hours_ago:
            current_hour = datetime.now().hour
            hour = (current_hour - int(raw_hours_ago.group(1))) % 24
            extracted['incident_hour_of_the_day'] = hour
            return

        # Fallback: direct hour pattern
        hour_val, _ = self.regex_extractor.extract('incident_hour_of_the_day', text)
        if hour_val:
            try:
                extracted['incident_hour_of_the_day'] = int(hour_val)
            except (ValueError, TypeError):
                pass

    def _normalize_police_report(self, extracted: Dict, text: str):
        """
        Populate police_report_available from the raw 'police_report' pattern key.
        The raw key 'police_report' is an internal-only field and will be removed
        from the output in the cleanup step.
        """
        if 'police_report_available' not in extracted:
            val, _ = self.regex_extractor.extract('police_report', text)
            if val:
                normalized = self.field_normalizer.normalize_yes_no(val)
                if normalized:
                    extracted['police_report_available'] = normalized
            else:
                # Plain "police report available" / "police report filed" in text
                if re.search(r'police\s+report\s+(?:available|filed|submitted)', text, re.IGNORECASE):
                    extracted['police_report_available'] = 'YES'
                elif re.search(r'no\s+police\s+report', text, re.IGNORECASE):
                    extracted['police_report_available'] = 'NO'

    def _normalize_amount_field(self, extracted: Dict, text: str):
        """Normalize total_claim_amount if not yet set"""
        if 'total_claim_amount' not in extracted:
            amount_val, _ = self.regex_extractor.extract('total_claim_amount', text)
            if amount_val:
                normalized = self.field_normalizer.normalize_amount(amount_val)
                if normalized:
                    extracted['total_claim_amount'] = normalized

    def _normalize_int_fields(self, extracted: Dict, text: str):
        """
        Convert integer fields to int.
        This handler is registered for each int field individually but
        the actual conversion is done in bulk in _post_process_fields (step 8).
        This method exists so the derived_fields dispatch table is consistent.
        """
        pass  # Bulk conversion handled in _post_process_fields step 8

    def _normalize_auto_year(self, extracted: Dict, text: str):
        """Ensure auto_year is a valid integer within a reasonable range"""
        if 'auto_year' in extracted:
            year = extracted['auto_year']
            try:
                year_int = int(str(year).replace(',', ''))
                if 1980 <= year_int <= datetime.now().year + 1:
                    extracted['auto_year'] = year_int
                else:
                    del extracted['auto_year']
            except (ValueError, TypeError):
                extracted.pop('auto_year', None)

    def _normalize_auto_make(self, extracted: Dict, text: str):
        """Ensure auto_make is normalized (FieldNormalizer handles the heavy lifting)"""
        if 'auto_make' in extracted:
            normalized = self.field_normalizer.normalize_auto_make(extracted['auto_make'])
            if normalized:
                extracted['auto_make'] = normalized

    def _normalize_date(self, extracted: Dict, text: str):
        """Normalize incident_date to YYYY-MM-DD"""
        if 'incident_date' in extracted:
            norm_date = self.field_normalizer.normalize_date(extracted['incident_date'])
            if norm_date:
                extracted['incident_date'] = norm_date

    # -------------------------------------------------------------------------
    # LLM fallback
    # -------------------------------------------------------------------------

    async def _extract_with_llm(self, text: str, missing_fields: List[str]) -> Dict[str, Any]:
        """Use LLM only for truly missing critical fields"""
        if not missing_fields:
            return {}

        try:
            import httpx
            import json

            prompt = f"""
Extract ONLY these specific fields from the insurance claim text.
If a field cannot be found, return null for that field.
Do NOT invent or guess values.

Fields needed: {', '.join(missing_fields)}

Claim text: {text[:2000]}

Return ONLY valid JSON with these fields:
{{
    "policy_number": "POL-XXXXXX or null",
    "customer_id": "CUST-XXXXX or null",
    "customer_name": "full name or null",
    "incident_date": "YYYY-MM-DD or null",
    "incident_type": "type or null",
    "auto_make": "make or null",
    "auto_model": "model or null",
    "auto_year": year_as_integer_or_null
}}
"""

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    llm_response = data.get('response', '{}')
                    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))

        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")

        return {}