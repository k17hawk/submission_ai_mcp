"""Agent 3: Risk Rule Checker – Loads rules from risk_rules.yaml"""

import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

from src.mcp_submission_parsing.config.logger_config import get_logger  
logger = get_logger('risk agent')

class RiskAgent:
    """Evaluates risk rules against claim data using configuration from YAML"""

    def __init__(self, config_path: str = None):
        """
        Initialize risk agent.
        
        Args:
            config_path: Path to risk_rules.yaml. If None, uses default location.
        """
        logger.info("Initializing RiskAgent")
        
        if config_path is None:
            # Try to find it relative to this file or in common config folder
            base_path = Path(__file__).parent.parent / 'config' / 'risk_rules.yaml'
            if base_path.exists():
                config_path = str(base_path)
                logger.debug(f"Found config file at: {config_path}")
            else:
                # Fallback to local directory
                config_path = 'risk_rules.yaml'
                logger.debug(f"No config found in default location, falling back to: {config_path}")
        
        self.config_path = config_path
        logger.info(f"Using config path: {self.config_path}")
        self.rules = []
        self.severity_weights = {}
        self.thresholds = {}
        logger.debug("Loading configuration from YAML")
        self._load_config()
        logger.info(f"RiskAgent initialization complete: {len(self.rules)} rules loaded")
    
    def _load_config(self):
        """Load rules and config from YAML file"""
        logger.info(f"Loading risk rules from {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.debug(f"YAML file loaded successfully")
            
            self.rules = config.get('rules', [])
            self.severity_weights = config.get('severity_weights', {
                'CRITICAL': 0.40, 'HIGH': 0.30, 'MEDIUM': 0.15, 'LOW': 0.07
            })
            self.thresholds = config.get('thresholds', {
                'siu_referral': 0.60,
                'adjuster_referral': 0.30,
                'auto_approve': 0.10,
                'auto_deny': 0.70
            })
            logger.info(f"Loaded {len(self.rules)} risk rules from {self.config_path}")
            logger.debug(f"Severity weights: {self.severity_weights}")
            logger.debug(f"Thresholds: {self.thresholds}")
            
            # Log rule IDs for debugging
            rule_ids = [rule.get('rule_id', 'UNKNOWN') for rule in self.rules]
            logger.debug(f"Loaded rule IDs: {rule_ids}")
            
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found at {self.config_path}: {e}")
            logger.warning("Falling back to default rules")
            self._load_default_rules()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {self.config_path}: {e}", exc_info=True)
            logger.warning("Falling back to default rules")
            self._load_default_rules()
        except Exception as e:
            logger.error(f"Failed to load risk rules: {e}", exc_info=True)
            logger.warning("Falling back to default rules")
            self._load_default_rules()
    
    def _load_default_rules(self):
        """Fallback hardcoded rules if YAML not found"""
        logger.warning("Loading fallback default rules")
        self.rules = [
            {
                'rule_id': 'R001',
                'name': 'Policy Must Be Active',
                'severity': 'CRITICAL',
                'conditions': [{'field': 'verification.is_active', 'operator': 'eq', 'value': True}],
                'action_on_fail': 'DENY',
                'fail_message': 'Policy is not active'
            },
            {
                'rule_id': 'R002',
                'name': 'Incident Within Policy Period',
                'severity': 'CRITICAL',
                'conditions': [{'field': 'verification.incident_in_policy_period', 'operator': 'eq', 'value': True}],
                'action_on_fail': 'DENY',
                'fail_message': 'Incident date outside policy period'
            }
        ]
        self.severity_weights = {'CRITICAL': 0.40, 'HIGH': 0.30, 'MEDIUM': 0.15, 'LOW': 0.07}
        self.thresholds = {'siu_referral': 0.60, 'adjuster_referral': 0.30, 'auto_approve': 0.10, 'auto_deny': 0.70}
        logger.warning(f"Using fallback default rules ({len(self.rules)} rules) - risk_rules.yaml not loaded properly")
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all risk rules against parsed claim and policy verification"""
        logger.info("Starting risk evaluation")
        logger.debug(f"Payload received: {payload.keys()}")
        
        parsed = payload.get('parsed', {})
        verification = payload.get('verification', {})
        
        logger.debug(f"Parsed data keys: {parsed.keys() if parsed else 'None'}")
        logger.debug(f"Verification data keys: {verification.keys() if verification else 'None'}")
        logger.debug(f"Verification contains: is_active={verification.get('is_active')}, incident_in_policy_period={verification.get('incident_in_policy_period')}")
        
        violations = []
        warnings = []
        risk_score = 0.0
        
        logger.info(f"Evaluating {len(self.rules)} risk rules")
        
        for idx, rule in enumerate(self.rules, 1):
            rule_id = rule.get('rule_id', f'UNKNOWN_{idx}')
            rule_name = rule.get('name', 'Unnamed Rule')
            severity = rule.get('severity', 'MEDIUM')
            conditions = rule.get('conditions', [])
            
            logger.debug(f"Evaluating rule {idx}/{len(self.rules)}: {rule_id} - {rule_name} (Severity: {severity})")
            logger.debug(f"Rule conditions: {conditions}")
            
            # Evaluate the rule: if any condition fails, rule is violated
            rule_passed = self._evaluate_rule(conditions, parsed, verification)
            logger.debug(f"Rule {rule_id} evaluation result: {'PASSED' if rule_passed else 'FAILED'}")
            
            if not rule_passed:
                severity = rule['severity']
                weight = self.severity_weights.get(severity, 0.10)
                risk_score += weight
                logger.info(f"Rule {rule_id} FAILED - Adding {weight} to risk score (new score: {risk_score:.3f})")
                
                violation = {
                    'rule_id': rule['rule_id'],
                    'rule_name': rule['name'],
                    'severity': severity,
                    'message': self._render_message(rule.get('fail_message', ''), parsed, verification),
                    'action': rule.get('action_on_fail', 'REVIEW')
                }
                
                logger.debug(f"Violation details: {violation}")
                
                if severity in ['CRITICAL', 'HIGH']:
                    violations.append(violation)
                    logger.debug(f"Added to violations (now {len(violations)} total)")
                else:
                    warnings.append(violation)
                    logger.debug(f"Added to warnings (now {len(warnings)} total)")
            else:
                logger.debug(f"Rule {rule_id} PASSED - No action needed")
        
        risk_score = min(risk_score, 1.0)
        logger.info(f"Final risk score after evaluation: {risk_score:.3f} (capped at 1.0)")
        
        # Determine risk level and pass/fail
        auto_deny_threshold = self.thresholds.get('auto_deny', 0.70)
        auto_approve_threshold = self.thresholds.get('auto_approve', 0.10)
        
        logger.debug(f"Thresholds - Auto Deny: {auto_deny_threshold}, Auto Approve: {auto_approve_threshold}")
        
        if risk_score >= auto_deny_threshold:
            risk_level = 'CRITICAL'
            passed = False
            logger.warning(f"Risk score {risk_score:.3f} >= {auto_deny_threshold} -> CRITICAL risk, NOT passed")
        elif risk_score >= 0.40:
            risk_level = 'HIGH'
            passed = False
            logger.warning(f"Risk score {risk_score:.3f} between 0.40 and {auto_deny_threshold} -> HIGH risk, NOT passed")
        elif risk_score >= 0.15:
            risk_level = 'MEDIUM'
            passed = True
            logger.info(f"Risk score {risk_score:.3f} between 0.15 and 0.40 -> MEDIUM risk, passed")
        else:
            risk_level = 'LOW'
            passed = True
            logger.info(f"Risk score {risk_score:.3f} < 0.15 -> LOW risk, passed")
        
        # Auto decision suggestions
        auto_decision = None
        if risk_score >= auto_deny_threshold:
            auto_decision = 'DENY'
            logger.info(f"Auto-decision triggered: DENY (risk score {risk_score:.3f} >= {auto_deny_threshold})")
        elif risk_score <= auto_approve_threshold and not violations:
            auto_decision = 'APPROVE'
            logger.info(f"Auto-decision triggered: APPROVE (risk score {risk_score:.3f} <= {auto_approve_threshold} and no violations)")
        else:
            logger.debug(f"No auto-decision triggered - risk_score={risk_score:.3f}, violations={len(violations)}")
        
        siu_threshold = self.thresholds.get('siu_referral', 0.60)
        adjuster_threshold = self.thresholds.get('adjuster_referral', 0.30)
        
        requires_siu = risk_score >= siu_threshold
        requires_adjuster = risk_score >= adjuster_threshold
        
        logger.info(f"SIU referral required: {requires_siu} (threshold={siu_threshold})")
        logger.info(f"Adjuster referral required: {requires_adjuster} (threshold={adjuster_threshold})")
        
        result = {
            'passed': passed,
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'violations': violations,
            'warnings': warnings,
            'requires_siu': requires_siu,
            'requires_adjuster': requires_adjuster,
            'auto_decision': auto_decision
        }
        
        logger.info(f"Risk evaluation complete - Result: passed={passed}, risk_level={risk_level}, score={risk_score:.3f}")
        logger.debug(f"Full result: {result}")
        
        return result
    
    def _evaluate_rule(self, conditions: List[Dict], parsed: Dict, verification: Dict) -> bool:
        """
        Evaluate a rule's conditions. 
        Supports 'any' wrapper for OR logic, otherwise all conditions must pass (AND).
        """
        if not conditions:
            logger.debug("No conditions for rule - returning True")
            return True
        
        # Check if the rule uses 'any' (logical OR)
        if len(conditions) == 1 and 'any' in conditions[0]:
            any_conditions = conditions[0]['any']
            logger.debug(f"Evaluating OR logic with {len(any_conditions)} conditions")
            # Rule passes if any of the inner conditions passes
            result = any(self._evaluate_condition(cond, parsed, verification) for cond in any_conditions)
            logger.debug(f"OR evaluation result: {result}")
            return result
        
        # Default: all conditions must pass (AND)
        logger.debug(f"Evaluating AND logic with {len(conditions)} conditions")
        result = all(self._evaluate_condition(cond, parsed, verification) for cond in conditions)
        logger.debug(f"AND evaluation result: {result}")
        return result
    
    def _evaluate_condition(self, cond: Dict, parsed: Dict, verification: Dict) -> bool:
        """Evaluate a single condition"""
        field_path = cond.get('field')
        operator = cond.get('operator')
        value = cond.get('value')
        value_field = cond.get('value_field')
        subfield = cond.get('subfield')   # used for 'not_contains'
        low = cond.get('low')
        high = cond.get('high')
        threshold = cond.get('threshold')
        
        logger.debug(f"Evaluating condition - field={field_path}, operator={operator}, value={value}, value_field={value_field}")
        
        # Get actual field value from parsed or verification
        actual = self._get_nested_value(field_path, parsed, verification)
        logger.debug(f"Actual value retrieved: {actual}")
        
        # Get comparison value (either literal or from another field)
        compare_value = value
        if value_field:
            compare_value = self._get_nested_value(value_field, parsed, verification)
            logger.debug(f"Comparison value from field {value_field}: {compare_value}")
        else:
            logger.debug(f"Comparison literal value: {compare_value}")
        
        # Operator implementations
        result = False
        if operator == 'eq':
            result = actual == compare_value
            logger.debug(f"Equality check: {actual} == {compare_value} -> {result}")
        elif operator == 'lte':
            result = self._safe_compare(actual, compare_value, lambda a, b: a <= b)
            logger.debug(f"LTE check: {actual} <= {compare_value} -> {result}")
        elif operator == 'gte':
            result = self._safe_compare(actual, compare_value, lambda a, b: a >= b)
            logger.debug(f"GTE check: {actual} >= {compare_value} -> {result}")
        elif operator == 'not_contains':
            # For list or string containment
            if actual is None:
                logger.debug("Actual value is None for not_contains - returning True")
                return True
            if subfield and isinstance(actual, dict):
                actual = actual.get(subfield, [])
                logger.debug(f"Extracted subfield '{subfield}': {actual}")
            if isinstance(actual, str):
                result = compare_value not in actual
                logger.debug(f"not_contains (string): '{compare_value}' not in '{actual}' -> {result}")
            elif isinstance(actual, list):
                result = compare_value not in actual
                logger.debug(f"not_contains (list): {compare_value} not in {actual} -> {result}")
            else:
                logger.debug(f"not_contains on unsupported type {type(actual)} - returning True")
                return True
        elif operator == 'days_after':
            # actual date should be <= (value_field date + threshold days)
            result = self._days_compare(actual, compare_value, threshold, after=True)
            logger.debug(f"days_after check: days between {actual} and {compare_value} with threshold {threshold} -> {result}")
        elif operator == 'days_before':
            # actual date should be >= (value_field date - threshold days)
            result = self._days_compare(actual, compare_value, threshold, after=False)
            logger.debug(f"days_before check: days between {actual} and {compare_value} with threshold {threshold} -> {result}")
        elif operator == 'not_between':
            # actual hour not between low and high (inclusive wrap-around for late night)
            result = self._not_between(actual, low, high)
            logger.debug(f"not_between check: {actual} not between {low} and {high} -> {result}")
        elif operator == 'is_weekday':
            # actual date should be weekday (Monday=0, Sunday=6, weekend is 5,6)
            result = self._is_weekday(actual)
            logger.debug(f"is_weekday check: {actual} is weekday -> {result}")
        else:
            logger.warning(f"Unsupported operator: {operator} - returning True (default pass)")
            return True  # Default to pass if unknown operator
        
        return result
    
    def _get_nested_value(self, path: str, parsed: Dict, verification: Dict) -> Any:
        """Get value from nested dict using dot notation, e.g., 'verification.is_active' or 'parsed.total_claim_amount'"""
        if not path:
            logger.debug("Empty path provided - returning None")
            return None
        
        parts = path.split('.')
        logger.debug(f"Getting nested value for path: {path} (parts: {parts})")
        
        if parts[0] == 'parsed':
            current = parsed
            logger.debug("Using parsed data dictionary")
        elif parts[0] == 'verification':
            current = verification
            logger.debug("Using verification data dictionary")
        else:
            # Assume it's a direct key in verification (backward compatible)
            logger.debug(f"Assuming direct key in verification: {path}")
            current = verification
            parts = [path]  # treat whole path as key
        
        for i, part in enumerate(parts[1:], 1):
            if isinstance(current, dict):
                current = current.get(part)
                logger.debug(f"Traversed to '{part}': {current}")
            else:
                logger.debug(f"Cannot traverse further - current is not dict at part '{part}' (type: {type(current)})")
                return None
        
        logger.debug(f"Final value for path '{path}': {current}")
        return current
    
    def _safe_compare(self, actual, compare, comp_func):
        """Safely compare values, converting to numeric if possible"""
        try:
            actual_num = float(actual) if actual is not None else None
            compare_num = float(compare) if compare is not None else None
            if actual_num is None or compare_num is None:
                logger.debug(f"Cannot compare - actual={actual}, compare={compare} (None values)")
                return False
            result = comp_func(actual_num, compare_num)
            logger.debug(f"Numeric comparison: {actual_num} vs {compare_num} -> {result}")
            return result
        except (ValueError, TypeError) as e:
            logger.warning(f"Error in numeric comparison: {e} (actual={actual}, compare={compare})")
            return False
    
    def _days_compare(self, actual, compare, threshold, after=True):
        """Compare dates: if after=True, actual <= compare + threshold days; else actual >= compare - threshold days"""
        try:
            actual_date = self._to_date(actual)
            compare_date = self._to_date(compare)
            if actual_date is None or compare_date is None:
                logger.debug(f"Cannot compare dates - actual={actual}, compare={compare} (invalid dates)")
                return False
            delta = (actual_date - compare_date).days
            logger.debug(f"Date comparison: {actual_date} - {compare_date} = {delta} days")
            if after:
                # Condition passes if incident date is more than threshold days after effective date?
                result = delta > threshold
                logger.debug(f"days_after check: {delta} > {threshold} -> {result}")
                return result
            else:
                # days_before: compare_date - actual_date > threshold
                result = (compare_date - actual_date).days > threshold
                logger.debug(f"days_before check: {compare_date} - {actual_date} = {(compare_date - actual_date).days} > {threshold} -> {result}")
                return result
        except Exception as e:
            logger.warning(f"Error in days comparison: {e}")
            return False
    
    def _to_date(self, value):
        """Convert string or datetime to date object"""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                result = datetime.strptime(value, '%Y-%m-%d').date()
                logger.debug(f"Converted string '{value}' to date: {result}")
                return result
            except ValueError as e:
                logger.debug(f"Failed to parse date string '{value}': {e}")
                pass
        return None
    
    def _not_between(self, actual, low, high):
        """Check if actual value (numeric) is NOT between low and high inclusive, handling wrap (e.g., 23-5)"""
        if actual is None:
            logger.debug("Actual value is None for not_between - returning True")
            return True
        try:
            val = int(actual)
            if low <= high:
                result = val < low or val > high
                logger.debug(f"not_between (normal range): {val} not between {low} and {high} -> {result}")
                return result
            else:
                # Wrap-around range: e.g., 23 to 5 means 23,0,1,2,3,4,5
                # Actually not_between: we want to return True if NOT in [low, high] wrap.
                # For late-night: low=23, high=5. The "bad" hours are 23,0,1,2,3,4,5.
                # So condition passes (not between) if hour is 6..22.
                if val >= low or val <= high:
                    result = False
                else:
                    result = True
                logger.debug(f"not_between (wrap range): {val} not between {low} and {high} (wrap) -> {result}")
                return result
        except Exception as e:
            logger.warning(f"Error in not_between check: {e}")
            return True
    
    def _is_weekday(self, actual):
        """Return True if actual date is Monday-Friday (weekday)"""
        dt = self._to_date(actual)
        if dt is None:
            logger.debug("Invalid date for is_weekday - returning True")
            return True
        # Monday=0, Sunday=6
        is_weekday = dt.weekday() < 5
        logger.debug(f"Date {dt} is weekday: {is_weekday} (weekday={dt.weekday()})")
        return is_weekday
    
    def _render_message(self, template: str, parsed: Dict, verification: Dict) -> str:
        """Replace placeholders like {parsed.field} and {verification.field} in message"""
        if not template:
            logger.debug("Empty message template")
            return ""
        
        result = template
        logger.debug(f"Rendering message template: {template}")
        
        # Simple replacement
        for key, value in parsed.items():
            if value is not None:
                placeholder = f'{{parsed.{key}}}'
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
                    logger.debug(f"Replaced {placeholder} with {value}")
        
        for key, value in verification.items():
            if value is not None:
                placeholder = f'{{verification.{key}}}'
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
                    logger.debug(f"Replaced {placeholder} with {value}")
        
        # Also handle {days_diff} which might be computed on the fly – you can add more sophisticated rendering if needed
        logger.debug(f"Final rendered message: {result}")
        return result