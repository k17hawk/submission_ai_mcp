"""Agent 1: Submission Parser"""

import re
from typing import Dict, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.models import ParsedClaim


class ParserAgent:
    """Parses claim submissions using regex and LLM"""
    
    def __init__(self):
        # Load patterns from config
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load regex patterns from config"""
        config_path = Path(__file__).parent.parent / "config" / "field_definitions.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('patterns', {})
        return {}
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process claim text and extract fields"""
        
        text = payload.get('text', '')
        use_llm = payload.get('use_llm', True)
        
        # Extract fields using regex
        parsed = self._extract_with_regex(text)
        
        # Enhance with LLM if requested
        if use_llm and text:
            llm_fields = await self._extract_with_llm(text)
            parsed.update(llm_fields)
            parsed['llm_enhanced'] = True
        
        # Calculate completeness
        required_fields = ['policy_number', 'incident_date', 'incident_type', 'auto_make']
        missing = [f for f in required_fields if not parsed.get(f)]
        parsed['missing_fields'] = missing
        
        return parsed
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract fields using regex patterns"""
        result = {}
        
        # Policy number
        pattern = self.patterns.get('policy_number', [r'POL-\d{6}'])
        match = re.search(pattern[0] if isinstance(pattern, list) else pattern, text, re.I)
        if match:
            result['policy_number'] = match.group(0)
        
        # Incident date
        date_pattern = self.patterns.get('incident_date', [r'\d{4}-\d{2}-\d{2}'])
        match = re.search(date_pattern[0] if isinstance(date_pattern, list) else date_pattern, text)
        if match:
            result['incident_date'] = match.group(0)
        
        # Vehicle info (year make model)
        vehicle_pattern = r'(\d{4})\s+(Toyota|Honda|Ford|Chevrolet|BMW|Nissan|Jeep|Dodge|Hyundai|Kia)\s+([A-Za-z0-9\-]+)'
        match = re.search(vehicle_pattern, text, re.I)
        if match:
            result['auto_year'] = int(match.group(1))
            result['auto_make'] = match.group(2).title()
            result['auto_model'] = match.group(3)
        
        # Claim amount
        amount_pattern = r'\$?([\d,]+\.?\d*)'
        matches = re.findall(amount_pattern, text)
        if matches:
            # Take the largest amount as total claim
            amounts = [float(m.replace(',', '')) for m in matches]
            result['total_claim_amount'] = max(amounts)
        
        # Incident type
        type_pattern = r'(Single Vehicle|Multi-vehicle|Parked Car|Vehicle Theft|Other)'
        match = re.search(type_pattern, text, re.I)
        if match:
            result['incident_type'] = match.group(1)
        
        return result
    
    async def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract fields that regex missed"""
        
        try:
            import httpx
            
            prompt = f"""
            Extract insurance claim fields from this text. Return ONLY JSON.
            
            Text: {text[:1000]}
            
            Extract:
            - policy_number (format POL-XXXXXX)
            - incident_date (YYYY-MM-DD)
            - incident_type
            - auto_make
            - auto_model
            - auto_year
            - total_claim_amount
            
            JSON:
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
                    import json
                    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                        
        except Exception as e:
            print(f"LLM extraction failed: {e}")
        
        return {}