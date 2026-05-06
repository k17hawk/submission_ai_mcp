"""
MCP Server 3: Risk Rule Checker (Config-Driven)
Loads rules from YAML. No hardcoded logic.
Evaluates conditions dynamically against parsed + verification data.
"""

import yaml
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RuleViolation:
    rule_id: str
    rule_name: str
    severity: str
    message: str
    action: str


@dataclass
class RiskAssessment:
    passed: bool
    risk_score: float
    risk_level: str
    violations: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    requires_siu: bool = False
    requires_adjuster: bool = False
    auto_decision: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITION EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ConditionEvaluator:
    """
    Evaluates conditions dynamically from config.
    Supports operators: eq, neq, gt, gte, lt, lte, contains, not_contains,
                        in, not_in, between, not_between, days_after, days_before,
                        is_weekday, is_weekend
    """
    
    def __init__(self):
        self.operators = {
            "eq": self._op_eq,
            "neq": self._op_neq,
            "gt": self._op_gt,
            "gte": self._op_gte,
            "lt": self._op_lt,
            "lte": self._op_lte,
            "contains": self._op_contains,
            "not_contains": self._op_not_contains,
            "in": self._op_in,
            "not_in": self._op_not_in,
            "between": self._op_between,
            "not_between": self._op_not_between,
            "days_after": self._op_days_after,
            "days_before": self._op_days_before,
            "is_weekday": self._op_is_weekday,
            "is_weekend": self._op_is_weekend,
        }
    
    def resolve_value(self, field_path: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a dotted path like 'parsed.total_claim_amount'
        against the context dict.
        """
        parts = field_path.split(".")
        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value
    
    def resolve(self, spec: Union[str, int, float, bool], context: Dict[str, Any]) -> Any:
        """
        If spec is a string starting with 'parsed.' or 'verification.', resolve it.
        Otherwise return the literal value.
        """
        if isinstance(spec, str) and ("." in spec):
            return self.resolve_value(spec, context)
        return spec
    
    def evaluate_condition(self, condition: Dict, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition dict."""
        # Handle 'all' / 'any' compound conditions
        if "all" in condition:
            return all(self.evaluate_condition(c, context) for c in condition["all"])
        if "any" in condition:
            return any(self.evaluate_condition(c, context) for c in condition["any"])
        
        field = condition.get("field")
        operator = condition.get("operator")
        
        # Get the actual value from context
        actual = self.resolve_value(field, context)
        
        # Get the expected value
        if "value_field" in condition:
            expected = self.resolve_value(condition["value_field"], context)
        elif "value" in condition:
            expected = condition["value"]
        else:
            expected = None
        
        # Get extra params
        threshold = condition.get("threshold")
        low = condition.get("low")
        high = condition.get("high")
        subfield = condition.get("subfield")
        
        op_func = self.operators.get(operator)
        if op_func:
            return op_func(actual, expected, threshold=threshold, low=low, high=high, subfield=subfield)
        
        return True  # Unknown operator → pass (don't block)
    
    # ── Operator implementations ───────────────────────────────────────────
    
    def _op_eq(self, actual, expected, **kwargs):
        return actual == expected
    
    def _op_neq(self, actual, expected, **kwargs):
        return actual != expected
    
    def _op_gt(self, actual, expected, **kwargs):
        if actual is None: return False
        return float(actual) > float(expected)
    
    def _op_gte(self, actual, expected, **kwargs):
        if actual is None: return False
        return float(actual) >= float(expected)
    
    def _op_lt(self, actual, expected, **kwargs):
        if actual is None: return False
        return float(actual) < float(expected)
    
    def _op_lte(self, actual, expected, **kwargs):
        if actual is None: return False
        return float(actual) <= float(expected)
    
    def _op_contains(self, actual, expected, **kwargs):
        if actual is None: return False
        subfield = kwargs.get("subfield")
        if isinstance(actual, list):
            if subfield:
                return any(subfield in str(item) for item in actual)
            return expected in actual
        return str(expected).lower() in str(actual).lower()
    
    def _op_not_contains(self, actual, expected, **kwargs):
        return not self._op_contains(actual, expected, **kwargs)
    
    def _op_in(self, actual, expected, **kwargs):
        if actual is None: return False
        return actual in expected
    
    def _op_not_in(self, actual, expected, **kwargs):
        if actual is None: return True
        return actual not in expected
    
    def _op_between(self, actual, expected, **kwargs):
        if actual is None: return False
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 0)
        return low <= float(actual) <= high
    
    def _op_not_between(self, actual, expected, **kwargs):
        if actual is None: return True
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 0)
        # Handle wrap-around like 23-5 (night hours)
        if low > high:
            return not (float(actual) >= low or float(actual) <= high)
        return not (low <= float(actual) <= high)
    
    def _op_days_after(self, actual, expected, **kwargs):
        """Check if actual date is MORE than threshold days after expected date."""
        if not actual or not expected: return True
        threshold = kwargs.get("threshold", 30)
        try:
            d1 = datetime.strptime(str(actual), "%Y-%m-%d")
            d2 = datetime.strptime(str(expected), "%Y-%m-%d")
            diff = (d1 - d2).days
            return diff > threshold
        except (ValueError, TypeError):
            return True
    
    def _op_days_before(self, actual, expected, **kwargs):
        """Check if actual date is MORE than threshold days before expected date."""
        if not actual or not expected: return True
        threshold = kwargs.get("threshold", 30)
        try:
            d1 = datetime.strptime(str(actual), "%Y-%m-%d")
            d2 = datetime.strptime(str(expected), "%Y-%m-%d")
            diff = (d2 - d1).days
            return diff > threshold
        except (ValueError, TypeError):
            return True
    
    def _op_is_weekday(self, actual, expected, **kwargs):
        if not actual: return True
        try:
            d = datetime.strptime(str(actual), "%Y-%m-%d")
            return d.weekday() < 5  # Mon-Fri
        except (ValueError, TypeError):
            return True
    
    def _op_is_weekend(self, actual, expected, **kwargs):
        if not actual: return False
        try:
            d = datetime.strptime(str(actual), "%Y-%m-%d")
            return d.weekday() >= 5  # Sat-Sun
        except (ValueError, TypeError):
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RiskRuleEngine:
    """
    Loads rules from YAML config.
    Evaluates each rule's conditions against the claim context.
    No hardcoded business logic — all rules are data.
    """
    
    def __init__(self, config_path: str = None):
        self.evaluator = ConditionEvaluator()
        self.rules = []
        self.severity_weights = {}
        self.thresholds = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load rules from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Rules config not found: {config_path}")
        
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        self.rules = config.get("rules", [])
        self.severity_weights = config.get("severity_weights", {
            "CRITICAL": 0.40, "HIGH": 0.30, "MEDIUM": 0.15, "LOW": 0.07
        })
        self.thresholds = config.get("thresholds", {
            "siu_referral": 0.60,
            "adjuster_referral": 0.30,
            "auto_approve": 0.10,
            "auto_deny": 0.70,
        })
        
        print(f"Loaded {len(self.rules)} rules from {config_path}")
    
    def evaluate(self, parsed: Dict[str, Any], verification: Dict[str, Any]) -> RiskAssessment:
        """Run all rules against the claim context."""
        context = {
            "parsed": parsed,
            "verification": verification,
        }
        
        violations = []
        warnings_list = []
        critical_count = 0
        high_count = 0
        
        for rule in self.rules:
            conditions = rule.get("conditions", [])
            
            # All conditions must pass for the rule to pass
            rule_passed = True
            for condition in conditions:
                if not self.evaluator.evaluate_condition(condition, context):
                    rule_passed = False
                    break
            
            if rule_passed:
                continue  # Rule passed — no violation
            
            # Rule failed — create violation
            message = self._render_message(rule.get("fail_message", ""), context)
            
            violation = RuleViolation(
                rule_id=rule.get("rule_id", "?"),
                rule_name=rule.get("name", "Unknown"),
                severity=rule.get("severity", "MEDIUM"),
                message=message,
                action=rule.get("action_on_fail", "NOTE"),
            )
            
            violation_dict = asdict(violation)
            
            if violation.severity == "CRITICAL":
                violations.append(violation_dict)
                critical_count += 1
            elif violation.severity == "HIGH":
                violations.append(violation_dict)
                high_count += 1
            elif violation.severity == "MEDIUM":
                violations.append(violation_dict)
            else:
                warnings_list.append(violation_dict)
        
        # ── Calculate risk score ───────────────────────────────────────────
        risk_score = 0.0
        for v in violations:
            risk_score += self.severity_weights.get(v["severity"], 0.10)
        for w in warnings_list:
            risk_score += self.severity_weights.get(w["severity"], 0.05)
        
        risk_score = min(round(risk_score, 3), 1.0)
        
        # ── Determine risk level ───────────────────────────────────────────
        if risk_score >= 0.70:
            risk_level = "CRITICAL"
        elif risk_score >= 0.40:
            risk_level = "HIGH"
        elif risk_score >= 0.15:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # ── Decisions ──────────────────────────────────────────────────────
        has_critical_or_high = critical_count > 0 or high_count > 0
        passed = not has_critical_or_high
        
        requires_siu = risk_score >= self.thresholds.get("siu_referral", 0.60)
        requires_adjuster = risk_score >= self.thresholds.get("adjuster_referral", 0.30)
        
        if has_critical_or_high and critical_count > 0:
            auto_decision = "DENY"
        elif risk_score <= self.thresholds.get("auto_approve", 0.10) and not has_critical_or_high:
            auto_decision = "APPROVE"
        else:
            auto_decision = None  # Needs human
        
        return RiskAssessment(
            passed=passed,
            risk_score=risk_score,
            risk_level=risk_level,
            violations=violations,
            warnings=warnings_list,
            requires_siu=requires_siu,
            requires_adjuster=requires_adjuster,
            auto_decision=auto_decision,
        )
    
    def _render_message(self, template: str, context: Dict) -> str:
        """Replace {parsed.field} and {verification.field} in message templates."""
        pattern = r"\{([a-zA-Z_.]+)\}"

        def replacer(match):
            path = match.group(1)

            # Special calculated values
            if path == "days_diff":
                return self._calc_days_diff(context)
            if path == "day_of_week":
                return self._calc_day_of_week(context)

            value = self.evaluator.resolve_value(path, context)

            # Handle None / missing fields
            if value is None or value == "":
                return "Unknown"

            if isinstance(value, float):
                # Check if there's a '$' immediately before the placeholder in the template
                has_dollar_prefix = template[max(0, match.start() - 1):match.start()] == "$"
                if has_dollar_prefix:
                    return f"{value:,.2f}"       # template already has "$"
                else:
                    return f"${value:,.2f}"      # renderer adds the "$"
            return str(value)

        return re.sub(pattern, replacer, template)
    
    def _calc_days_diff(self, context: Dict) -> str:
        """Calculate days between incident and policy dates for messages."""
        inc = context.get("parsed", {}).get("incident_date")
        eff = context.get("verification", {}).get("effective_date")
        exp = context.get("verification", {}).get("expiration_date")
        try:
            d_inc = datetime.strptime(str(inc), "%Y-%m-%d")
            d_eff = datetime.strptime(str(eff), "%Y-%m-%d")
            d_exp = datetime.strptime(str(exp), "%Y-%m-%d")
            from_start = (d_inc - d_eff).days
            from_end = (d_exp - d_inc).days
            if from_start < 60:
                return str(from_start)
            return str(from_end)
        except:
            return "?"
    
    def _calc_day_of_week(self, context: Dict) -> str:
        inc = context.get("parsed", {}).get("incident_date")
        try:
            d = datetime.strptime(str(inc), "%Y-%m-%d")
            return d.strftime("%A")
        except:
            return "?"


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

# Load engine from config
CONFIG_PATH = Path(__file__).parent / "config" / "risk_rules.yaml"
engine = RiskRuleEngine(str(CONFIG_PATH))


def check_risk_rules(
    parsed: Dict[str, Any],
    verification: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MCP Tool: Evaluate all risk rules against a claim.
    
    Args:
        parsed: Output from Agent 1 (parse_submission)
        verification: Output from Agent 2 (verify_policy_for_claim)
    
    Returns:
        Risk assessment with violations, risk score, and decision guidance.
    """
    assessment = engine.evaluate(parsed, verification)
    return asdict(assessment)


def reload_rules(config_path: str = None) -> Dict[str, Any]:
    """
    MCP Tool: Reload rules from config (for hot-reloading without restart).
    """
    path = config_path or CONFIG_PATH
    engine.load_config(str(path))
    return {"status": "reloaded", "rule_count": len(engine.rules)}


def get_rule_definitions() -> List[Dict[str, str]]:
    """MCP Tool: Get all current rule definitions."""
    return [
        {
            "rule_id": r.get("rule_id"),
            "name": r.get("name"),
            "description": r.get("description", ""),
            "severity": r.get("severity"),
            "action_on_fail": r.get("action_on_fail"),
        }
        for r in engine.rules
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parsed = {
        "policy_number": "POL-760113",
        "incident_date": "2020-08-17",
        "incident_type": "Multi-vehicle Collision",
        "incident_severity": "Major Damage",
        "total_claim_amount": 4520.00,
        "auto_make": "Honda",
        "auto_model": "CR-V",
        "auto_year": 2008,
        "witnesses": 1,
        "police_report_available": "YES",
        "number_of_vehicles_involved": 2,
        "incident_hour_of_the_day": 14,
    }
    
    verification = {
        "found": True,
        "is_active": True,
        "policy_status": "Active",
        "incident_in_policy_period": True,
        "effective_date": "2019-06-01",
        "expiration_date": "2021-06-01",
        "coverage_code": "COLL",
        "coverage_name": "Collision",
        "coverage_limit": 250000,
        "deductible": 500,
        "prior_claims": 1,
        "errors": [],
        "warnings": [],
    }
    
    assessment = check_risk_rules(parsed, verification)
    
    print("=" * 60)
    print("CONFIG-DRIVEN RISK RULE CHECKER — TEST")
    print("=" * 60)
    print(f"Rules loaded: {len(engine.rules)}")
    print(f"Passed: {assessment['passed']}")
    print(f"Risk Score: {assessment['risk_score']} ({assessment['risk_level']})")
    print(f"Auto Decision: {assessment['auto_decision']}")
    print(f"Needs SIU: {assessment['requires_siu']}")
    print(f"Needs Adjuster: {assessment['requires_adjuster']}")
    
    if assessment['violations']:
        print(f"\nViolations ({len(assessment['violations'])}):")
        for v in assessment['violations']:
            print(f"  [{v['severity']}] {v['rule_id']} {v['rule_name']}")
            print(f"    → {v['message']}")
    
    if assessment['warnings']:
        print(f"\nWarnings ({len(assessment['warnings'])}):")
        for w in assessment['warnings']:
            print(f"  [{w['severity']}] {w['rule_name']}")
    
    print(f"\nRule definitions available: {len(get_rule_definitions())}")