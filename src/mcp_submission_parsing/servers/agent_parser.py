"""Agent 1: Submission Parser - Config-driven using regex extractor and field normalizer"""

import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_submission_parsing.extractor.field_normalizer import FieldNormalizer
from src.mcp_submission_parsing.extractor.regex_extractor import RegexExtractor
from src.mcp_submission_parsing.config import load_config, get_patterns, get_normalizer_config
from src.mcp_submission_parsing.config.logger_config import get_logger

logger = get_logger("parsing agent")


class ParserAgent:
    """
    Parses claim submissions using:
    - RegexExtractor (patterns from field_definitions.yaml)
    - FieldNormalizer (synonyms, valid values, date/amount cleaning)
    - Optional LLM enhancement for missing critical fields
    """

    def __init__(self):
        logger.info("Initializing ParserAgent")

        try:
            self.config = load_config()
            patterns = get_patterns()
            normalizer_config = get_normalizer_config()
            logger.debug(f"Loaded {len(patterns)} patterns from config")
            logger.debug(f"Loaded normalizer config with {len(normalizer_config.get('field_synonyms', {}))} field synonyms")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise

        self.regex_extractor = RegexExtractor(patterns)
        self.field_normalizer = FieldNormalizer(normalizer_config)
        self.internal_only_fields = {
            'police_report',       
            'year_make_model',     
            'make_model',          
            'city_state',
        }
        logger.debug(f"Internal-only fields: {self.internal_only_fields}")

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
        logger.debug(f"Registered {len(self.derived_fields)} derived field handlers")

        self.pattern_fields = list(patterns.keys())
        logger.info(f"ParserAgent initialized successfully with {len(self.pattern_fields)} pattern fields")

    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process claim text and extract fields - config-driven"""
        logger.info("Starting claim submission parsing")
        logger.debug(f"Payload keys: {list(payload.keys())}")

        text = payload.get('text', '')
        use_llm = payload.get('use_llm', False)
        force_llm_enhancement = payload.get('force_llm_enhancement', False)  
        submission_time: datetime = payload.get('submission_time', datetime.now())

        logger.debug(f"Text length: {len(text)} characters")
        logger.debug(f"Use LLM: {use_llm}")
        logger.debug(f"Force LLM Enhancement: {force_llm_enhancement}")
        logger.debug(f"Submission time: {submission_time}")

        if not text:
            logger.warning("No claim text provided in payload")
            return {
                'error': 'No claim text provided',
                'missing_fields': ['claim_text'],
                'extraction_confidence': 0.0
            }

        logger.info("Step 1: Extracting fields using regex patterns")
        extracted = self._extract_all_fields(text)
        logger.debug(f"Regex extraction complete. Extracted {len(extracted)} fields: {list(extracted.keys())}")

        logger.info("Step 2: Applying post-processing and normalization")
        extracted = self._post_process_fields(extracted, text, submission_time)
        logger.debug(f"Post-processing complete. Final fields: {list(extracted.keys())}")
        required_fields = self.config.get('required_fields', [
            'policy_number', 'incident_date', 'incident_type', 'auto_make'
        ])
        missing = [f for f in required_fields if not extracted.get(f)]
        logger.info(f"Required fields: {required_fields}")
        logger.info(f"Missing required fields: {missing}")

        scoreable_fields = [f for f in self.pattern_fields if f not in self.internal_only_fields]
        total_possible = len(scoreable_fields)
        extracted_count = sum(1 for f in scoreable_fields if extracted.get(f))
        confidence = extracted_count / total_possible if total_possible > 0 else 0.0
        logger.info(f"Confidence calculation: {extracted_count}/{total_possible} fields extracted = {confidence:.2%}")

        result = {
            **extracted,
            'extraction_confidence': round(confidence, 2),
            'missing_fields': missing,
            'llm_enhanced': False,
            'extraction_success': len(missing) == 0,
            'extracted_fields_count': extracted_count,
            'total_possible_fields': total_possible,
        }
        logger.debug(f"Initial result: extraction_success={result['extraction_success']}, confidence={result['extraction_confidence']}")

        # Step 5: LLM enhancement for missing critical fields (optional)
        # Check if we should use LLM: either we have missing fields OR we're forcing LLM enhancement
        should_use_llm = use_llm and (missing or force_llm_enhancement)
        
        if should_use_llm:
            # Determine which fields to target with LLM
            if force_llm_enhancement:
                # When forcing, enhance all critical fields plus some useful ones
                target_fields = list(set(required_fields + [
                    'auto_model', 'auto_year', 'customer_name', 'customer_id', 
                    'incident_location', 'collision_type', 'number_of_vehicles'
                ]))
                logger.info(f"Step 5: Forcing LLM enhancement for critical fields: {target_fields}")
            else:
                # Normal mode: only enhance missing fields
                target_fields = missing
                logger.info(f"Step 5: Attempting LLM enhancement for missing fields: {missing}")
            
            llm_fields = await self._extract_with_llm(text, target_fields)
            logger.debug(f"LLM returned fields: {list(llm_fields.keys())}")
            
            llm_filled_count = 0
            llm_overrode_count = 0
            
            for key, value in llm_fields.items():
                if value:
                    existing_value = extracted.get(key)
                    
                    # If field is missing OR we're forcing LLM and value is different/better
                    if not existing_value:
                        extracted[key] = value
                        result[key] = value
                        if key in missing:
                            missing.remove(key)
                        llm_filled_count += 1
                        logger.info(f"LLM filled missing field '{key}' with value '{value}'")
                    elif force_llm_enhancement and str(existing_value) != str(value):
                        extracted[key] = value
                        result[key] = value
                        llm_overrode_count += 1
                        logger.info(f"LLM overrode field '{key}': '{existing_value}' -> '{value}'")

            result['llm_enhanced'] = True
            result['llm_filled_count'] = llm_filled_count
            result['llm_overrode_count'] = llm_overrode_count
            result['missing_fields'] = missing
            result['extraction_confidence'] = min(0.95, confidence + 0.2)
            result['extraction_success'] = len(missing) == 0
            
            logger.info(f"After LLM enhancement: filled {llm_filled_count} fields, overrode {llm_overrode_count} fields, missing={missing}, new confidence={result['extraction_confidence']}")
            logger.debug(f"Final result details: {result}")

        logger.info(f"Parsing complete. Success: {result['extraction_success']}, Confidence: {result['extraction_confidence']}")
        return result


    def _extract_all_fields(self, text: str) -> Dict[str, Any]:
        """Extract all fields defined in patterns using RegexExtractor"""
        logger.debug("Starting regex extraction for all fields")
        # Always derived via custom logic — never taken raw from regex
        always_derived = {'incident_hour_of_the_day'}
        result = {}
        
        for field_name in self.pattern_fields:
            if field_name in always_derived:
                logger.debug(f"Skipping derived field '{field_name}' from regex extraction")
                continue
            value, confidence = self.regex_extractor.extract(field_name, text)
            if value is not None:
                result[field_name] = value
                logger.debug(f"Extracted '{field_name}': '{value}' (confidence: {confidence})")
            else:
                logger.debug(f"No match for field '{field_name}'")
                
        logger.info(f"Regex extraction found {len(result)} fields")
        return result

    # -------------------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------------------

    def _post_process_fields(self, extracted: Dict[str, Any], text: str, submission_time: datetime = None) -> Dict[str, Any]:
        """
        Apply field-specific normalisation and derived field logic.
        Uses FieldNormalizer where possible, plus custom handlers.
        """
        logger.debug("Starting post-processing of extracted fields")
        if submission_time is None:
            submission_time = datetime.now()
            logger.debug("Using current time as submission_time")
            
        processed = extracted.copy()

        # ------------------------------------------------------------------
        # 0. Pileup / multi-vehicle detection from raw text.
        # ------------------------------------------------------------------
        if not processed.get('incident_type') or processed.get('incident_type', '').isdigit():
            logger.debug("Checking for pileup/multi-vehicle pattern in text")
            if re.search(
                r'\d+[-\s]?(?:car|vehicle)s?\s+(?:pileup|collision|accident|crash)',
                text, re.IGNORECASE
            ):
                processed['incident_type'] = 'Multi-vehicle Collision'
                logger.info(f"Detected multi-vehicle incident, set incident_type to '{processed['incident_type']}'")

        # ------------------------------------------------------------------
        # 1. Standard normalisation using FieldNormalizer
        # ------------------------------------------------------------------
        logger.debug("Applying standard normalization using FieldNormalizer")
        for field, value in list(processed.items()):
            if field in self.internal_only_fields:
                logger.debug(f"Skipping normalization for internal field '{field}'")
                continue

            normalized = None
            if field == 'incident_date':
                normalized = self.field_normalizer.normalize_date(value)
                logger.debug(f"Normalized incident_date: '{value}' -> '{normalized}'")
            elif field in ('total_claim_amount', 'property_damage_amount'):
                normalized = self.field_normalizer.normalize_amount(value)
                logger.debug(f"Normalized amount '{field}': '{value}' -> '{normalized}'")
            elif field in ('police_report_available', 'property_damage'):
                normalized = self.field_normalizer.normalize_yes_no(value)
                logger.debug(f"Normalized yes/no '{field}': '{value}' -> '{normalized}'")
            elif field == 'auto_make':
                normalized = self.field_normalizer.normalize_auto_make(value)
                logger.debug(f"Normalized auto_make: '{value}' -> '{normalized}'")
            elif field in ('incident_type', 'incident_severity', 'incident_location',
                        'authorities_contacted', 'collision_type'):
                normalized = self.field_normalizer.normalize(field, value)
                logger.debug(f"Normalized '{field}': '{value}' -> '{normalized}'")

            if normalized is not None:
                processed[field] = normalized
                logger.debug(f"Updated field '{field}' to normalized value '{normalized}'")

        # ------------------------------------------------------------------
        # 2. Special / derived fields
        # ------------------------------------------------------------------
        logger.debug("Processing derived fields")
        for derived_field, handler in self.derived_fields.items():
            if derived_field == 'incident_hour_of_the_day':
                logger.debug(f"Calling handler for derived field '{derived_field}'")
                handler(processed, text, submission_time)
                continue
            if derived_field in processed and processed[derived_field] is not None:
                logger.debug(f"Derived field '{derived_field}' already has value '{processed[derived_field]}', skipping")
                continue
            logger.debug(f"Calling handler for derived field '{derived_field}'")
            handler(processed, text, submission_time)

        # ------------------------------------------------------------------
        # 3. Policy number prefix cleanup (ensure POL-XXXXXX)
        # ------------------------------------------------------------------
        if 'policy_number' in processed:
            original = processed['policy_number']
            pn = processed['policy_number']
            pn = re.sub(r'^POL[-\s]?', '', pn, flags=re.IGNORECASE)
            processed['policy_number'] = f"POL-{pn}"
            logger.debug(f"Cleaned policy_number: '{original}' -> '{processed['policy_number']}'")

        # ------------------------------------------------------------------
        # 4. Customer ID prefix cleanup
        # ------------------------------------------------------------------
        if 'customer_id' in processed:
            original = processed['customer_id']
            cid = processed['customer_id']
            if not cid.upper().startswith('CUST'):
                processed['customer_id'] = f"CUST-{cid}"
                logger.debug(f"Added CUST- prefix to customer_id: '{original}' -> '{processed['customer_id']}'")
            else:
                cid = re.sub(r'^CUST[-\s]?', '', cid, flags=re.IGNORECASE)
                processed['customer_id'] = f"CUST-{cid}"
                logger.debug(f"Reformatted customer_id: '{original}' -> '{processed['customer_id']}'")

        # ------------------------------------------------------------------
        # 5. Customer name: strip trailing context words
        # ------------------------------------------------------------------
        if 'customer_name' in processed:
            original = processed['customer_name']
            name = processed['customer_name']
            name = re.sub(r'\s+(?:with|CUST[-\w]*|policy[-\w]*).*$', '', name, flags=re.IGNORECASE)
            processed['customer_name'] = name.strip().rstrip(',;:')
            logger.debug(f"Cleaned customer_name: '{original}' -> '{processed['customer_name']}'")

        # ------------------------------------------------------------------
        # 6. incident_state: must be exactly a 2-letter uppercase state code.
        # ------------------------------------------------------------------
        state = processed.get('incident_state', '')
        valid_states = set(self.config.get('valid_values', {}).get('incident_state', []))
        if not re.fullmatch(r'[A-Z]{2}', str(state)) or (valid_states and state not in valid_states):
            logger.debug(f"Invalid incident_state '{state}', attempting to extract from text")
            state_match = re.search(
                r'\b(A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|'
                r'N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AT]|W[AVIY])\b', text
            )
            if state_match:
                processed['incident_state'] = state_match.group(1)
                logger.debug(f"Extracted incident_state from text: '{state_match.group(1)}'")
            else:
                processed.pop('incident_state', None)
                logger.debug("No valid incident_state found in text")

        # ------------------------------------------------------------------
        # 7. incident_location: removed the override that set raw highway code.
        #    Now relies solely on FieldNormalizer mapping (e.g., I-95 -> Interstate).
        # ------------------------------------------------------------------
        # The old override block has been removed. The normalized value from step 1 remains.

        # ------------------------------------------------------------------
        # 8. incident_city: strip trailing conjunctions/noise.
        # ------------------------------------------------------------------
        if 'incident_city' in processed:
            original = processed['incident_city']
            city = processed['incident_city']
            city = re.sub(r'\s+(?:and|the|or|at|in|near)\b.*$', '', city, flags=re.IGNORECASE).strip().rstrip(',;:')
            processed['incident_city'] = city if len(city) >= 2 else processed.pop('incident_city', None)
            logger.debug(f"Cleaned incident_city: '{original}' -> '{processed.get('incident_city', 'REMOVED')}'")

        # ------------------------------------------------------------------
        # 9. auto_model: drop noise values that aren't real model names.
        # ------------------------------------------------------------------
        if 'auto_model' in processed:
            model = processed['auto_model']
            noise = {'was involved', 'involved', 'the vehicle', 'a vehicle'}
            if model.lower() in noise or len(model.split()) > 3 or not re.match(r'^[A-Z]', model):
                processed.pop('auto_model', None)
                logger.debug(f"Removed invalid auto_model '{model}' (noise or invalid format)")

        # ------------------------------------------------------------------
        # 10. total_claim_amount: ensure it's a float, not a raw string.
        # ------------------------------------------------------------------
        if 'total_claim_amount' in processed:
            val = processed['total_claim_amount']
            if not isinstance(val, float):
                logger.debug(f"Converting total_claim_amount from {type(val)} to float")
                normalized = self.field_normalizer.normalize_amount(str(val))
                if normalized is not None:
                    processed['total_claim_amount'] = normalized
                    logger.debug(f"Normalized total_claim_amount: '{val}' -> '{normalized}'")
                else:
                    processed.pop('total_claim_amount', None)
                    logger.debug(f"Failed to normalize total_claim_amount '{val}', removed field")

        # ------------------------------------------------------------------
        # 11. police_report_available: normalize raw match to YES / NO / ?
        # ------------------------------------------------------------------
        if 'police_report_available' in processed:
            val = processed['police_report_available']
            if val not in ('YES', 'NO', '?'):
                logger.debug(f"Normalizing police_report_available from '{val}'")
                normalized = self.field_normalizer.normalize_yes_no(str(val))
                if normalized:
                    processed['police_report_available'] = normalized
                    logger.debug(f"Normalized to '{normalized}'")
                elif re.search(r'police\s+report\s+(?:available|filed|submitted)', str(val), re.IGNORECASE):
                    processed['police_report_available'] = 'YES'
                    logger.debug("Set to 'YES' based on pattern match")
                elif re.search(r'no\s+police\s+report', str(val), re.IGNORECASE):
                    processed['police_report_available'] = 'NO'
                    logger.debug("Set to 'NO' based on pattern match")

        # ------------------------------------------------------------------
        # 12. Integer fields: ensure witnesses, bodily_injuries,
        #     number_of_vehicles are stored as int, not string.
        # ------------------------------------------------------------------
        for int_field in ('witnesses', 'bodily_injuries', 'number_of_vehicles'):
            if int_field in processed:
                try:
                    original = processed[int_field]
                    processed[int_field] = int(str(processed[int_field]).replace(',', ''))
                    logger.debug(f"Converted {int_field} from '{original}' to {processed[int_field]}")
                except (ValueError, TypeError):
                    logger.debug(f"Failed to convert {int_field} '{processed[int_field]}', removing field")
                    processed.pop(int_field, None)

        # ------------------------------------------------------------------
        # 13. Remove internal-only pattern keys from the final output.
        # ------------------------------------------------------------------
        removed_fields = []
        for field in self.internal_only_fields:
            if field in processed:
                removed_fields.append(field)
                processed.pop(field, None)
        if removed_fields:
            logger.debug(f"Removed internal-only fields: {removed_fields}")

        logger.info("Post-processing complete")
        return processed


    def _normalize_auto_year(self, extracted: Dict, text: str, submission_time=None):
        """Ensure auto_year is a valid integer, preferring vehicle-specific year over date year."""
        # If year_make_model exists, use that as the primary source
        if 'year_make_model' in extracted and extracted['year_make_model']:
            try:
                year = int(str(extracted['year_make_model']).strip())
                current_year = datetime.now().year
                if 1980 <= year <= current_year + 1:
                    extracted['auto_year'] = year
                    logger.debug(f"Using auto_year from year_make_model: {year}")
                    # Remove the raw auto_year if it conflicts
                    if 'auto_year' in extracted:
                        del extracted['auto_year']
                    return
            except (ValueError, TypeError):
                pass
        
        # Otherwise, fall back to existing auto_year with date conflict check
        if 'auto_year' in extracted:
            year = extracted['auto_year']
            try:
                year_int = int(str(year).replace(',', ''))
                current_year = datetime.now().year
                
                # Reject if this year matches the incident_date year
                inc_date = extracted.get('incident_date')
                if inc_date and isinstance(inc_date, str) and len(inc_date) >= 4:
                    date_year = int(inc_date[:4])
                    if year_int == date_year:
                        logger.debug(f"Auto_year {year_int} matches incident_date year, discarding")
                        del extracted['auto_year']
                        return
                
                if 1980 <= year_int <= current_year + 1:
                    extracted['auto_year'] = year_int
                else:
                    del extracted['auto_year']
            except (ValueError, TypeError):
                extracted.pop('auto_year', None)

    # -------------------------------------------------------------------------
    # Derived field handlers
    # -------------------------------------------------------------------------

    def _derive_incident_hour(self, extracted: Dict, text: str, submission_time: datetime = None):
        """
        Derive incident_hour_of_the_day from:
        1. "X hours ago" phrasing  → submission_time.hour - X (mod 24)
        2. Explicit time like "2:30 PM" in incident_time
        3. Direct hour pattern
        """
        logger.debug("Deriving incident_hour_of_the_day")
        if submission_time is None:
            submission_time = datetime.now()
            logger.debug("Using current time as submission_time")

        # Try incident_time pattern first
        time_val, _ = self.regex_extractor.extract('incident_time', text)
        if time_val:
            logger.debug(f"Found incident_time: '{time_val}'")
            hours_ago = re.search(r'(\d+)\s*hours?\s+ago', time_val, re.IGNORECASE)
            if hours_ago:
                hour = (submission_time.hour - int(hours_ago.group(1))) % 24
                extracted['incident_hour_of_the_day'] = hour
                logger.info(f"Derived hour from 'X hours ago': {hours_ago.group(1)} hours ago -> hour {hour}")
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
                logger.info(f"Derived hour from explicit time: {time_match.group(0)} -> hour {hour}")
                return

        # Check raw text for "X hours ago"
        raw_hours_ago = re.search(r'(\d+)\s*hours?\s+ago', text, re.IGNORECASE)
        if raw_hours_ago:
            hour = (submission_time.hour - int(raw_hours_ago.group(1))) % 24
            extracted['incident_hour_of_the_day'] = hour
            logger.info(f"Derived hour from raw text 'hours ago': {raw_hours_ago.group(1)} hours ago -> hour {hour}")
            return

        # Fallback: direct hour pattern
        hour_val, _ = self.regex_extractor.extract('incident_hour_of_the_day', text)
        if hour_val:
            try:
                extracted['incident_hour_of_the_day'] = int(hour_val)
                logger.info(f"Derived hour from direct pattern: '{hour_val}' -> hour {extracted['incident_hour_of_the_day']}")
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to convert hour value '{hour_val}': {e}")

        if 'incident_hour_of_the_day' not in extracted:
            logger.debug("Could not derive incident_hour_of_the_day")

    def _normalize_police_report(self, extracted: Dict, text: str, submission_time=None):
        """
        Populate police_report_available from the raw 'police_report' pattern key.
        The raw key 'police_report' is an internal-only field and will be removed
        from the output in the cleanup step.
        """
        logger.debug("Normalizing police_report_available")
        if 'police_report_available' not in extracted:
            val, _ = self.regex_extractor.extract('police_report', text)
            if val:
                logger.debug(f"Found police_report pattern: '{val}'")
                normalized = self.field_normalizer.normalize_yes_no(val)
                if normalized:
                    extracted['police_report_available'] = normalized
                    logger.info(f"Set police_report_available to '{normalized}' from pattern")
            else:
                # Plain "police report available" / "police report filed" in text
                if re.search(r'police\s+report\s+(?:available|filed|submitted)', text, re.IGNORECASE):
                    extracted['police_report_available'] = 'YES'
                    logger.info("Set police_report_available to 'YES' based on text pattern")
                elif re.search(r'no\s+police\s+report', text, re.IGNORECASE):
                    extracted['police_report_available'] = 'NO'
                    logger.info("Set police_report_available to 'NO' based on text pattern")
                else:
                    logger.debug("No police_report pattern found")

    def _normalize_amount_field(self, extracted: Dict, text: str, submission_time=None):
        """Normalize total_claim_amount if not yet set"""
        logger.debug("Normalizing total_claim_amount")
        if 'total_claim_amount' not in extracted:
            amount_val, _ = self.regex_extractor.extract('total_claim_amount', text)
            if amount_val:
                logger.debug(f"Found total_claim_amount pattern: '{amount_val}'")
                normalized = self.field_normalizer.normalize_amount(amount_val)
                if normalized:
                    extracted['total_claim_amount'] = normalized
                    logger.info(f"Set total_claim_amount to '{normalized}'")
                else:
                    logger.debug(f"Failed to normalize amount '{amount_val}'")
            else:
                logger.debug("No total_claim_amount pattern found")

    def _normalize_int_fields(self, extracted: Dict, text: str, submission_time=None):
        """
        Convert integer fields to int.
        This handler is registered for each int field individually but
        the actual conversion is done in bulk in _post_process_fields (step 8).
        This method exists so the derived_fields dispatch table is consistent.
        """
        logger.debug("Integer field normalization handler called (bulk conversion handled elsewhere)")
        pass  # Bulk conversion handled in _post_process_fields step 8

    def _normalize_auto_make(self, extracted: Dict, text: str, submission_time=None):
        """Ensure auto_make is normalized (FieldNormalizer handles the heavy lifting)"""
        logger.debug("Normalizing auto_make")
        if 'auto_make' in extracted:
            original = extracted['auto_make']
            normalized = self.field_normalizer.normalize_auto_make(extracted['auto_make'])
            if normalized:
                extracted['auto_make'] = normalized
                logger.debug(f"Normalized auto_make: '{original}' -> '{normalized}'")
            else:
                logger.debug(f"Failed to normalize auto_make '{original}'")

    def _normalize_date(self, extracted: Dict, text: str, submission_time=None):
        """Normalize incident_date to YYYY-MM-DD"""
        logger.debug("Normalizing incident_date")
        if 'incident_date' in extracted:
            original = extracted['incident_date']
            norm_date = self.field_normalizer.normalize_date(extracted['incident_date'])
            if norm_date:
                extracted['incident_date'] = norm_date
                logger.debug(f"Normalized incident_date: '{original}' -> '{norm_date}'")
            else:
                logger.debug(f"Failed to normalize incident_date '{original}'")

    # -------------------------------------------------------------------------
    # LLM fallback
    # -------------------------------------------------------------------------

    async def _extract_with_llm(self, text: str, missing_fields: List[str]) -> Dict[str, Any]:
        """Use LLM only for truly missing critical fields"""
        logger.info(f"Attempting LLM extraction for missing fields: {missing_fields}")
        if not missing_fields:
            logger.debug("No missing fields, skipping LLM extraction")
            return {}

        try:
            import httpx
            import json

            prompt = f"""
    Extract ONLY these specific fields from the insurance claim text.
    IMPORTANT: If a field is already present in the text, extract it exactly as written.
    For incident_type, if multiple types are mentioned, prioritize the most specific one.

    Fields needed: {', '.join(missing_fields)}

    Claim text: {text[:2000]}

    Return ONLY valid JSON with these fields (use null if not found):
    {{
        "policy_number": "POL-XXXXXX or null",
        "customer_id": "CUST-XXXXX or null", 
        "customer_name": "full name or null",
        "incident_date": "YYYY-MM-DD or null",
        "incident_type": "type (e.g., Front Collision, Rear-End, Multi-vehicle Collision) or null",
        "auto_make": "make or null",
        "auto_model": "model or null",
        "auto_year": year_as_integer_or_null,
        "collision_type": "specific collision type or null",
        "incident_location": "location or null"
    }}

    Return ONLY valid JSON, no other text.
    """
            logger.debug(f"LLM prompt prepared, text length: {len(text)} chars")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info("Calling LLM API at http://localhost:11434/api/generate")
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}  # Low temperature for consistent results
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    llm_response = data.get('response', '{}')
                    logger.debug(f"LLM response received, length: {len(llm_response)} chars")
                    
                    # Try to extract JSON from response
                    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                        logger.info(f"LLM extraction successful, returned {len(result)} fields")
                        logger.debug(f"LLM extracted values: {result}")
                        return result
                    else:
                        logger.warning("No JSON found in LLM response")
                        logger.debug(f"Raw LLM response: {llm_response}")
                else:
                    logger.error(f"LLM API returned status code {response.status_code}")

        except httpx.TimeoutException:
            logger.error("LLM API timeout after 30 seconds")
        except httpx.ConnectError:
            logger.error("Failed to connect to LLM API at localhost:11434")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)

        logger.warning(f"LLM extraction failed for fields: {missing_fields}")
        return {}