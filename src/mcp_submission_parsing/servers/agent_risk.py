"""Agent 3: Risk Rule Checker – Loads rules from risk_rules.yaml"""

import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskAgent:
    """Evaluates risk rules against claim data using configuration from YAML"""

    def __init__(self, config_path: str = None):
        """
        Initialize risk agent.
        
        Args:
            config_path: Path to risk_rules.yaml. If None, uses default location.
        """
        if config_path is None:
            # Try to find it relative to this file or in common config folder
            base_path = Path(__file__).parent.parent / 'config' / 'risk_rules.yaml'
            if base_path.exists():
                config_path = str(base_path)
            else:
                # Fallback to local directory
                config_path = 'risk_rules.yaml'
        
        self.config_path = config_path
        self.rules = []
        self.severity_weights = {}
        self.thresholds = {}
        self._load_config()
    
    def _load_config(self):
        """Load rules and config from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
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
        except Exception as e:
            logger.error(f"Failed to load risk rules: {e}", exc_info=True)
            # Fallback to minimal default rules
            self._load_default_rules()
    
    def _load_default_rules(self):
        """Fallback hardcoded rules if YAML not found"""
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
        logger.warning("Using fallback default rules (risk_rules.yaml not loaded)")
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all risk rules against parsed claim and policy verification"""
        parsed = payload.get('parsed', {})
        verification = payload.get('verification', {})
        
        violations = []
        warnings = []
        risk_score = 0.0
        
        for rule in self.rules:
            conditions = rule.get('conditions', [])
            # Evaluate the rule: if any condition fails, rule is violated
            rule_passed = self._evaluate_rule(conditions, parsed, verification)
            
            if not rule_passed:
                severity = rule['severity']
                risk_score += self.severity_weights.get(severity, 0.10)
                
                violation = {
                    'rule_id': rule['rule_id'],
                    'rule_name': rule['name'],
                    'severity': severity,
                    'message': self._render_message(rule.get('fail_message', ''), parsed, verification),
                    'action': rule.get('action_on_fail', 'REVIEW')
                }
                
                if severity in ['CRITICAL', 'HIGH']:
                    violations.append(violation)
                else:
                    warnings.append(violation)
        
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level and pass/fail
        if risk_score >= self.thresholds.get('auto_deny', 0.70):
            risk_level = 'CRITICAL'
            passed = False
        elif risk_score >= 0.40:
            risk_level = 'HIGH'
            passed = False
        elif risk_score >= 0.15:
            risk_level = 'MEDIUM'
            passed = True
        else:
            risk_level = 'LOW'
            passed = True
        
        # Auto decision suggestions
        auto_decision = None
        if risk_score >= self.thresholds.get('auto_deny', 0.70):
            auto_decision = 'DENY'
        elif risk_score <= self.thresholds.get('auto_approve', 0.10) and not violations:
            auto_decision = 'APPROVE'
        
        return {
            'passed': passed,
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'violations': violations,
            'warnings': warnings,
            'requires_siu': risk_score >= self.thresholds.get('siu_referral', 0.60),
            'requires_adjuster': risk_score >= self.thresholds.get('adjuster_referral', 0.30),
            'auto_decision': auto_decision
        }
    
    def _evaluate_rule(self, conditions: List[Dict], parsed: Dict, verification: Dict) -> bool:
        """
        Evaluate a rule's conditions. 
        Supports 'any' wrapper for OR logic, otherwise all conditions must pass (AND).
        """
        if not conditions:
            return True
        
        # Check if the rule uses 'any' (logical OR)
        if len(conditions) == 1 and 'any' in conditions[0]:
            any_conditions = conditions[0]['any']
            # Rule passes if any of the inner conditions passes
            return any(self._evaluate_condition(cond, parsed, verification) for cond in any_conditions)
        
        # Default: all conditions must pass (AND)
        return all(self._evaluate_condition(cond, parsed, verification) for cond in conditions)
    
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
        
        # Get actual field value from parsed or verification
        actual = self._get_nested_value(field_path, parsed, verification)
        
        # Get comparison value (either literal or from another field)
        compare_value = value
        if value_field:
            compare_value = self._get_nested_value(value_field, parsed, verification)
        
        # Operator implementations
        if operator == 'eq':
            return actual == compare_value
        elif operator == 'lte':
            return self._safe_compare(actual, compare_value, lambda a, b: a <= b)
        elif operator == 'gte':
            return self._safe_compare(actual, compare_value, lambda a, b: a >= b)
        elif operator == 'not_contains':
            # For list or string containment
            if actual is None:
                return True
            if subfield and isinstance(actual, dict):
                actual = actual.get(subfield, [])
            if isinstance(actual, str):
                return compare_value not in actual
            elif isinstance(actual, list):
                return compare_value not in actual
            else:
                return True
        elif operator == 'days_after':
            # actual date should be <= (value_field date + threshold days)
            return self._days_compare(actual, compare_value, threshold, after=True)
        elif operator == 'days_before':
            # actual date should be >= (value_field date - threshold days)
            return self._days_compare(actual, compare_value, threshold, after=False)
        elif operator == 'not_between':
            # actual hour not between low and high (inclusive wrap-around for late night)
            return self._not_between(actual, low, high)
        elif operator == 'is_weekday':
            # actual date should be weekday (Monday=0, Sunday=6, weekend is 5,6)
            return self._is_weekday(actual)
        else:
            logger.warning(f"Unsupported operator: {operator}")
            return True  # Default to pass if unknown operator
    
    def _get_nested_value(self, path: str, parsed: Dict, verification: Dict) -> Any:
        """Get value from nested dict using dot notation, e.g., 'verification.is_active' or 'parsed.total_claim_amount'"""
        if not path:
            return None
        parts = path.split('.')
        if parts[0] == 'parsed':
            current = parsed
        elif parts[0] == 'verification':
            current = verification
        else:
            # Assume it's a direct key in verification (backward compatible)
            current = verification
            parts = [path]  # treat whole path as key
        
        for part in parts[1:]:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
    
    def _safe_compare(self, actual, compare, comp_func):
        """Safely compare values, converting to numeric if possible"""
        try:
            actual_num = float(actual) if actual is not None else None
            compare_num = float(compare) if compare is not None else None
            if actual_num is None or compare_num is None:
                return False
            return comp_func(actual_num, compare_num)
        except (ValueError, TypeError):
            return False
    
    def _days_compare(self, actual, compare, threshold, after=True):
        """Compare dates: if after=True, actual <= compare + threshold days; else actual >= compare - threshold days"""
        try:
            actual_date = self._to_date(actual)
            compare_date = self._to_date(compare)
            if actual_date is None or compare_date is None:
                return False
            delta = (actual_date - compare_date).days
            if after:
                # Condition passes if incident date is more than threshold days after start? Wait, check R005: condition is "days_after" with threshold 30, but the rule logic in YAML says: conditions: - field: parsed.incident_date operator: days_after value_field: verification.effective_date threshold: 30. And action_on_fail: INVESTIGATE. That means if incident_date is within 30 days AFTER effective_date, that is a fail (because it's flagged for investigation). So the condition should pass if days_after > threshold? Actually the rule condition is evaluated; if it fails, the rule is violated. The condition in YAML says: field: parsed.incident_date operator: days_after value_field: verification.effective_date threshold: 30. The intended meaning: "incident date is more than 30 days after effective date" → condition passes (true) → no violation. If incident date is within 30 days, condition fails → violation. So we need to implement days_after as: (actual_date - compare_date).days > threshold. Similarly days_before: (compare_date - actual_date).days > threshold.
                return delta > threshold
            else:
                # days_before: compare_date - actual_date > threshold
                return (compare_date - actual_date).days > threshold
        except Exception:
            return False
    
    def _to_date(self, value):
        """Convert string or datetime to date object"""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return datetime.strptime(value, '%Y-%m-%d').date()
            except ValueError:
                pass
        return None
    
    def _not_between(self, actual, low, high):
        """Check if actual value (numeric) is NOT between low and high inclusive, handling wrap (e.g., 23-5)"""
        if actual is None:
            return True
        try:
            val = int(actual)
            if low <= high:
                return val < low or val > high
            else:
                # Wrap-around range: e.g., 23 to 5 means 23,0,1,2,3,4,5
                return val < low and val > high  # In wrap case, it's between if val >= low or val <= high
                # Actually not_between: we want to return True if NOT in [low, high] wrap. Let's simplify.
                # For late-night: low=23, high=5. The "bad" hours are 23,0,1,2,3,4,5.
                # So condition passes (not between) if hour is 6..22.
                if val >= low or val <= high:
                    return False  # it IS between (bad)
                return True
        except:
            return True
    
    def _is_weekday(self, actual):
        """Return True if actual date is Monday-Friday (weekday)"""
        dt = self._to_date(actual)
        if dt is None:
            return True
        # Monday=0, Sunday=6
        return dt.weekday() < 5
    
    def _render_message(self, template: str, parsed: Dict, verification: Dict) -> str:
        """Replace placeholders like {parsed.field} and {verification.field} in message"""
        if not template:
            return ""
        result = template
        # Simple replacement
        for key, value in parsed.items():
            if value is not None:
                result = result.replace(f'{{parsed.{key}}}', str(value))
        for key, value in verification.items():
            if value is not None:
                result = result.replace(f'{{verification.{key}}}', str(value))
        # Also handle {days_diff} which might be computed on the fly – you can add more sophisticated rendering if needed
        return result