"""Agent 1: Submission Parser - Enhanced with comprehensive extraction"""

import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import sys


class ParserAgent:
    """Parses claim submissions using regex patterns only (no dummy values)"""

    def __init__(self):
        self.patterns = {
            'policy_number': [
                r'Policy\s+(POL[-\s]?\d{6})',         # "Policy POL-651065"
                r'POL[-](\d{6})',                       # "POL-651065" → capture digits only
                r'policy[:\s#]+(POL[-]?\d{6})',
            ],
            'customer_id': [
                r'CUST[-](\d{5,})',                     # "CUST-93810" → digits only
                r'customer[:\s#]+(CUST[-]?\d+)',
            ],
            'customer_name': [
                r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s+with|\s+CUST|\s+policy|$)',
                r'^([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+with|\s+CUST)',
            ],
            'incident_date': [
                r'at\s+(\d{4}-\d{2}-\d{2})',           # prefer "at 2022-08-09"
                r'on\s+(\d{4}-\d{2}-\d{2})',
                r'\bon\s+(\d{1,2}/\d{1,2}/\d{4})',
            ],
            'incident_time': [
                r'(\d{1,2})\s*(?:hours?|hrs?)\s+ago',
                r'(?:at|around)\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))',
            ],
            'incident_type': [
                r'(\d+)[-\s]?(?:car|vehicle)\s+(pileup|collision|accident|crash)',
                r'(Single\s+Vehicle|Multi-vehicle|Parked\s+Car|Vehicle\s+Theft)',
                r'(rear|front|side)[-\s]?(?:end\s+collision|collision|impact)',
            ],
            'collision_type': [
                r'(Front|Rear|Side)\s+(?:Collision|End|Impact)',
                r'(Head-On|T-Bone|Sideswipe|Rear-End)',
            ],
            'incident_location': [
                r'(I-\d+)',
                r'(Highway\s*\d+|Interstate\s*\d+)',
                r'(Parking\s+Lot|Local\s+Road|Residential\s+Street|Freeway)',
            ],
            'incident_state': [
                r'\b(A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AT]|W[AVIY])\b'
            ],
            'incident_city': [
                r'(?:at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:[A-Z]{2})\b',
            ],
            'auto_year': [
                # Must NOT be immediately preceded by a digit (avoids grabbing from dates)
                r'(?<!\d)(19[8-9]\d|20[0-1]\d)(?!\d)',
            ],
            'auto_make': [
                r'\b(Toyota|Honda|Ford|Chevrolet|BMW|Nissan|Jeep|Dodge|Hyundai|Kia|Subaru|Tesla|Mercedes|Audi)\b',
            ],
            'auto_model': [
                r'(?:Chevrolet|Ford|Toyota|Honda|Nissan|Jeep|Dodge|Hyundai|Kia|Subaru|BMW|Audi|Mercedes)\s+([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+)?)',
            ],
            'number_of_vehicles': [
                r'(\d+)[-\s]?(?:car|vehicle)s?\s+(?:pileup|collision|accident|crash|involved)',
                r'(\d+)\s+vehicles?\s+involved',
            ],
            'witnesses': [
                r'(\d+)\s+witnesses?',
                r'witness(?:es)?\s*:?\s*(\d+)',
            ],
            'police_report': [
                r'police\s+report\s+(available|filed|yes|true)',
                r'(?:no|without)\s+police\s+report',
            ],
            'total_claim_amount': [
                # Must have $ or explicit "damage/claim" keyword to avoid grabbing random numbers
                r'\$\s*([\d,]+(?:\.\d{2})?)',
                r'(?:damage|claim|amount)[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)',
            ],
            'bodily_injuries': [
                r'(\d+)\s+(?:people|persons?|individuals?)\s+(?:injured|hurt)',
                r'(\d+)\s+injur(?:y|ies)',
            ],
            'property_damage': [
                r'property\s+damage\s+(?:estimated|around|about)?\s*\$?\s*([\d,]+)',
            ],
        }

    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process claim text and extract fields - no dummy values"""

        text = payload.get('text', '')
        use_llm = payload.get('use_llm', False)

        if not text:
            return {
                'error': 'No claim text provided',
                'missing_fields': ['claim_text'],
                'extraction_confidence': 0.0
            }

        extracted = self._extract_all_fields(text)

        required_fields = ['policy_number', 'incident_date', 'incident_type', 'auto_make']
        missing = [f for f in required_fields if not extracted.get(f)]

        total_possible = len(self.patterns)
        extracted_count = sum(1 for v in extracted.values() if v is not None and v != '')
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

    def _extract_all_fields(self, text: str) -> Dict[str, Any]:
        result = {}
        for field_name, patterns in self.patterns.items():
            value = self._extract_field(text, patterns)
            if value is not None:
                result[field_name] = value
        result = self._post_process_extractions(result, text)
        return result

    def _extract_field(self, text: str, patterns: List[str]) -> Optional[Any]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1) if match.lastindex else match.group(0)
                return value.strip()
        return None

    def _post_process_extractions(self, extracted: Dict, text: str) -> Dict:

        # ── policy_number: always add POL- prefix ──────────────────────────
        if 'policy_number' in extracted:
            pn = extracted['policy_number'].strip()
            # Strip any accidental "POL " or "POL-" duplication then re-add
            pn = re.sub(r'^POL[-\s]?', '', pn, flags=re.IGNORECASE)
            extracted['policy_number'] = f"POL-{pn}"

        # ── customer_id: always add CUST- prefix ───────────────────────────
        if 'customer_id' in extracted:
            cid = extracted['customer_id'].strip()
            cid = re.sub(r'^CUST[-\s]?', '', cid, flags=re.IGNORECASE)
            extracted['customer_id'] = f"CUST-{cid}"

        # ── customer_name: strip trailing junk ─────────────────────────────
        if 'customer_name' in extracted:
            name = extracted['customer_name'].strip()
            # Remove trailing "with", "CUST...", policy refs
            name = re.sub(r'\s+(with|CUST\S*|policy\S*).*$', '', name, flags=re.IGNORECASE)
            extracted['customer_name'] = name.strip()

        # ── auto_year: prefer vehicle year, not date year ──────────────────
        # Find ALL 4-digit year-like numbers not inside a date (YYYY-MM-DD)
        # Remove date strings first, then find years
        text_no_dates = re.sub(r'\d{4}-\d{2}-\d{2}', '', text)
        year_match = re.search(r'\b(19[8-9]\d|20[0-1]\d)\b', text_no_dates)
        if year_match:
            extracted['auto_year'] = int(year_match.group(1))
        elif 'auto_year' in extracted:
            # If what we got looks like it came from a date, drop it
            del extracted['auto_year']

        # ── incident_date: take the FIRST date in text (most likely incident) ─
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
        if dates:
            extracted['incident_date'] = dates[0]

        # ── total_claim_amount: clean and convert ──────────────────────────
        if 'total_claim_amount' in extracted:
            amount_str = str(extracted['total_claim_amount']).replace(',', '')
            try:
                extracted['total_claim_amount'] = float(amount_str)
            except ValueError:
                del extracted['total_claim_amount']

        # ── numeric fields ─────────────────────────────────────────────────
        for field in ['number_of_vehicles', 'witnesses', 'bodily_injuries']:
            if field in extracted:
                try:
                    extracted[field] = int(extracted[field])
                except (ValueError, TypeError):
                    pass

        # ── police report ──────────────────────────────────────────────────
        if 'police_report' in extracted:
            val = extracted['police_report'].lower()
            extracted['police_report_available'] = 'YES' if val in ['available', 'filed', 'yes', 'true'] else 'NO'
            del extracted['police_report']

        # ── incident_type: normalise to human-readable string ──────────────
        if 'incident_type' in extracted:
            raw = extracted['incident_type'].lower()
            if 'pileup' in raw or 'multi' in raw or re.search(r'^\d+', raw):
                extracted['incident_type'] = 'Multi-vehicle Collision'
            elif 'single' in raw:
                extracted['incident_type'] = 'Single Vehicle Collision'
            elif 'parked' in raw:
                extracted['incident_type'] = 'Parked Car'
            elif 'theft' in raw or 'stolen' in raw:
                extracted['incident_type'] = 'Vehicle Theft'
            elif 'rear' in raw:
                extracted['incident_type'] = 'Rear-End Collision'
            elif 'front' in raw:
                extracted['incident_type'] = 'Front Collision'
            elif 'side' in raw:
                extracted['incident_type'] = 'Side Collision'

        # ── incident_hour from "X hours ago" ──────────────────────────────
        if 'incident_time' in extracted:
            time_str = extracted['incident_time']
            hours_ago = re.search(r'(\d+)\s*hours?\s+ago', time_str, re.IGNORECASE)
            if hours_ago:
                current_hour = datetime.now().hour
                extracted['incident_hour_of_the_day'] = max(0, (current_hour - int(hours_ago.group(1))) % 24)
            else:
                time_match = re.search(r'(\d{1,2})(?::\d{2})?\s*(AM|PM)', time_str, re.IGNORECASE)
                if time_match:
                    hour = int(time_match.group(1))
                    if time_match.group(2).upper() == 'PM' and hour != 12:
                        hour += 12
                    elif time_match.group(2).upper() == 'AM' and hour == 12:
                        hour = 0
                    extracted['incident_hour_of_the_day'] = hour

        return extracted

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