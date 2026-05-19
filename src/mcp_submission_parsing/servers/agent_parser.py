"""Agent 1: Submission Parser - Enhanced with comprehensive extraction"""

import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import sys


class ParserAgent:
    """Parses claim submissions using regex patterns only (no dummy values)"""
    
    def __init__(self):
        # Comprehensive patterns for claim extraction
        self.patterns = {
            'policy_number': [
                r'POL[-\s]?(\d{6})',
                r'policy[:\s#]+([A-Z0-9\-]+)',
                r'#([A-Z0-9\-]{6,})'
            ],
            'customer_id': [
                r'CUST[-\s]?(\d{5,})',
                r'customer[:\s#]+([A-Z0-9\-]+)',
                r'ID[:\s]+([A-Z0-9\-]+)'
            ],
            'customer_name': [
                r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:with|policy)'
            ],
            'incident_date': [
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(?:on|at)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})'
            ],
            'incident_time': [
                r'(\d{1,2})\s*(?:hours?|hrs?)\s+(?:ago|later)',
                r'(?:at|around)\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))'
            ],
            'incident_type': [
                r'(\d+)[-\s]?(?:car|vehicle)\s+(pileup|collision|accident|crash)',
                r'(Single Vehicle|Multi-vehicle|Parked Car|Vehicle Theft)',
                r'(?:rear|front|side)[-\s]?(?:end|collision|impact)'
            ],
            'collision_type': [
                r'(Front|Rear|Side)\s+(?:Collision|End|Impact)',
                r'(Head-On|T-Bone|Sideswipe|Rear-End)'
            ],
            'incident_location': [
                r'(I-\d+|Highway|Interstate)\s*\d*',
                r'(Parking Lot|Local Road|Residential Street|Freeway)'
            ],
            'incident_state': [
                r'\b(A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AT]|W[AVIY])\b'
            ],
            'incident_city': [
                r'(?:at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:[A-Z]{2}|FL)'
            ],
            'auto_year': [
                r'\b(19[8-9]\d|20[0-2]\d)\b'
            ],
            'auto_make': [
                r'\b(Toyota|Honda|Ford|Chevrolet|BMW|Nissan|Jeep|Dodge|Hyundai|Kia|Subaru|Tesla|Mercedes|Audi)\b'
            ],
            'auto_model': [
                r'(?:Chevrolet|Ford|Toyota|Honda)\s+([A-Z][a-z]+(?:\s?[A-Z]?\d*)?)'
            ],
            'number_of_vehicles': [
                r'(\d+)[-\s]?(?:car|vehicle)s?',
                r'(\d+)\s+vehicles? involved'
            ],
            'witnesses': [
                r'(\d+)\s+witnesses?',
                r'witness(?:es)?\s*:?\s*(\d+)'
            ],
            'police_report': [
                r'police\s+report\s+(available|filed|yes|true)',
                r'(?:no|without)\s+police\s+report'
            ],
            'total_claim_amount': [
                r'\$?([\d,]+(?:\.\d{2})?)\s*(?:dollars?|USD)?',
                r'(?:damage|claim|amount)[:\s]+\$?([\d,]+)'
            ],
            'bodily_injuries': [
                r'(\d+)\s+(?:people|persons?|individuals?)\s+(?:injured|hurt)',
                r'(\d+)\s+injuries?'
            ],
            'property_damage': [
                r'property\s+damage\s+(?:estimated|around|about)?\s*\$?([\d,]+)'
            ]
        }
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process claim text and extract fields - no dummy values"""
        
        text = payload.get('text', '')
        use_llm = payload.get('use_llm', False)  # Default to False - regex only
        
        if not text:
            return {
                'error': 'No claim text provided',
                'missing_fields': ['claim_text'],
                'extraction_confidence': 0.0
            }
        
        # Extract fields using regex
        extracted = self._extract_all_fields(text)
        
        # Calculate what's missing
        required_fields = ['policy_number', 'incident_date', 'incident_type', 'auto_make']
        missing = [f for f in required_fields if not extracted.get(f)]
        
        # Calculate confidence based on extraction rate
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
            'total_possible_fields': total_possible
        }
        
        # Only use LLM if explicitly requested and regex failed for critical fields
        if use_llm and missing and len(missing) > 0:
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
        """Extract all fields using regex patterns"""
        result = {}
        
        for field_name, patterns in self.patterns.items():
            value = self._extract_field(text, patterns)
            if value is not None:
                result[field_name] = value
        
        # Post-process specific fields
        result = self._post_process_extractions(result, text)
        
        return result
    
    def _extract_field(self, text: str, patterns: List[str]) -> Optional[Any]:
        """Extract a single field using multiple patterns"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return first capture group if exists, otherwise full match
                value = match.group(1) if match.lastindex else match.group(0)
                return value.strip()
        return None
    
    def _post_process_extractions(self, extracted: Dict, text: str) -> Dict:
        """Post-process and validate extracted fields"""
        
        # Calculate incident hour if time mentioned
        if 'incident_time' in extracted and not extracted.get('incident_hour_of_the_day'):
            time_str = extracted['incident_time']
            if 'hours ago' in time_str.lower():
                hours = re.search(r'(\d+)', time_str)
                if hours:
                    # Rough estimate: assume claim filed at current time (14:00 for example)
                    # In production, use actual current time
                    current_hour = 14  # 2 PM assumption
                    hour_ago = current_hour - int(hours.group(1))
                    extracted['incident_hour_of_the_day'] = max(0, hour_ago % 24)
            else:
                # Parse actual time like "2 PM"
                time_match = re.search(r'(\d{1,2})\s*(?:AM|PM|am|pm)', time_str)
                if time_match:
                    hour = int(time_match.group(1))
                    if 'PM' in time_str.upper() and hour != 12:
                        hour += 12
                    elif 'AM' in time_str.upper() and hour == 12:
                        hour = 0
                    extracted['incident_hour_of_the_day'] = hour
        
        # Clean claim amount
        if 'total_claim_amount' in extracted:
            amount_str = extracted['total_claim_amount'].replace(',', '')
            try:
                extracted['total_claim_amount'] = float(amount_str)
            except:
                pass
        
        # Convert numeric fields
        numeric_fields = ['number_of_vehicles', 'witnesses', 'bodily_injuries', 'auto_year']
        for field in numeric_fields:
            if field in extracted:
                try:
                    extracted[field] = int(extracted[field])
                except:
                    pass
        
        # Normalize police report
        if 'police_report' in extracted:
            val = extracted['police_report'].lower()
            extracted['police_report_available'] = 'YES' if val in ['available', 'filed', 'yes', 'true'] else 'NO'
            del extracted['police_report']
        
        # Build full incident type
        if 'incident_type' in extracted:
            incident = extracted['incident_type'].lower()
            if 'pileup' in incident or 'multi' in incident:
                extracted['incident_type'] = 'Multi-vehicle Collision'
            elif 'single' in incident:
                extracted['incident_type'] = 'Single Vehicle Collision'
            elif 'parked' in incident:
                extracted['incident_type'] = 'Parked Car'
            elif 'theft' in incident or 'stolen' in incident:
                extracted['incident_type'] = 'Vehicle Theft'
        
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
                "auto_year": year or null
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
                    
                    # Extract JSON from response
                    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                        
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
        
        return {}