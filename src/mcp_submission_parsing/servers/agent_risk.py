"""Agent 3: Risk Rule Checker"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import yaml
import sys


class RiskAgent:
    """Evaluates risk rules against claim data"""
    
    def __init__(self, config_path: str = None):
        self.rules = []
        self.severity_weights = {
            'CRITICAL': 0.40,
            'HIGH': 0.30,
            'MEDIUM': 0.15,
            'LOW': 0.07
        }
        
        if config_path:
            self._load_rules(config_path)
        else:
            self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default risk rules"""
        self.rules = [
            {
                'rule_id': 'R001',
                'name': 'Policy Must Be Active',
                'severity': 'CRITICAL',
                'condition': 'verification.is_active == True',
                'message': 'Policy is not active'
            },
            {
                'rule_id': 'R002',
                'name': 'Claim Within Coverage Limit',
                'severity': 'HIGH',
                'condition': 'parsed.total_claim_amount <= verification.coverage_limit',
                'message': 'Claim amount exceeds coverage limit'
            },
            {
                'rule_id': 'R003',
                'name': 'Police Report Required for Large Claims',
                'severity': 'MEDIUM',
                'condition': 'parsed.total_claim_amount <= 5000 or parsed.police_report_available == "YES"',
                'message': 'Large claim without police report'
            },
            {
                'rule_id': 'R004',
                'name': 'Late Night Incident',
                'severity': 'MEDIUM',
                'condition': 'parsed.incident_hour_of_the_day not in [23,0,1,2,3,4,5]',
                'message': 'Incident occurred during late night hours'
            }
        ]
    
    def _load_rules(self, config_path: str):
        """Load rules from YAML config"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.rules = config.get('rules', [])
                self.severity_weights = config.get('severity_weights', self.severity_weights)
        except Exception as e:
            print(f"Error loading rules: {e}")
            self._load_default_rules()
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all risk rules"""
        
        parsed = payload.get('parsed', {})
        verification = payload.get('verification', {})
        
        violations = []
        warnings = []
        risk_score = 0.0
        
        for rule in self.rules:
            # Evaluate condition
            if not self._evaluate_condition(rule.get('condition', ''), parsed, verification):
                # Rule violated
                violation = {
                    'rule_id': rule['rule_id'],
                    'rule_name': rule['name'],
                    'severity': rule['severity'],
                    'message': self._render_message(rule['message'], parsed, verification),
                    'action': rule.get('action_on_fail', 'REVIEW')
                }
                
                severity = rule['severity']
                risk_score += self.severity_weights.get(severity, 0.10)
                
                if severity in ['CRITICAL', 'HIGH']:
                    violations.append(violation)
                else:
                    warnings.append(violation)
        
        # Determine final assessment
        risk_score = min(risk_score, 1.0)
        
        if risk_score >= 0.70:
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
        
        return {
            'passed': passed,
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'violations': violations,
            'warnings': warnings,
            'requires_siu': risk_score >= 0.60,
            'requires_adjuster': risk_score >= 0.30,
            'auto_decision': 'DENY' if risk_score >= 0.70 else ('APPROVE' if risk_score <= 0.10 else None)
        }
    
    def _evaluate_condition(self, condition: str, parsed: Dict, verification: Dict) -> bool:
        """Safely evaluate condition string"""
        try:
            # Create evaluation context
            context = {
                'parsed': parsed,
                'verification': verification
            }
            
            # Simple condition evaluation (in production, use proper expression parser)
            if '==' in condition:
                left, right = condition.split('==')
                left_val = self._get_nested_value(left.strip(), context)
                right_val = self._parse_value(right.strip())
                return left_val == right_val
            elif '<=' in condition:
                left, right = condition.split('<=')
                left_val = self._get_nested_value(left.strip(), context)
                right_val = self._parse_value(right.strip())
                return left_val <= right_val
            elif 'not in' in condition:
                left, right = condition.split('not in')
                left_val = self._get_nested_value(left.strip(), context)
                right_val = eval(right.strip())
                return left_val not in right_val
            
            return True
            
        except Exception as e:
            print(f"Condition evaluation error: {e}")
            return True
    
    def _get_nested_value(self, path: str, context: Dict) -> Any:
        """Get value from nested dictionary using dot notation"""
        parts = path.split('.')
        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse value string to appropriate type"""
        value_str = value_str.strip()
        
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        elif value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]
        elif value_str == 'True':
            return True
        elif value_str == 'False':
            return False
        elif value_str == 'None':
            return None
        elif value_str.isdigit():
            return int(value_str)
        elif value_str.replace('.', '').isdigit():
            return float(value_str)
        else:
            return value_str
    
    def _render_message(self, template: str, parsed: Dict, verification: Dict) -> str:
        """Render message template with values"""
        message = template
        # Simple template replacement
        for key, value in parsed.items():
            if value is not None:
                message = message.replace(f'{{parsed.{key}}}', str(value))
        
        for key, value in verification.items():
            if value is not None:
                message = message.replace(f'{{verification.{key}}}', str(value))
        
        return message