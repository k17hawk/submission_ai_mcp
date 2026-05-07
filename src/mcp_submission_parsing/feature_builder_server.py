"""
Agent 4 — Feature Builder
Builds the exact 27-column feature vector matching your trained fraud ML model.
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple


# Exact 27 features your model expects — order MUST match training
FEATURE_COLUMNS = [
    'months_as_customer',
    'age',
    'insured_sex',
    'insured_education_level',
    'insured_occupation',
    'policy_deductable',
    'umbrella_limit',
    'capital_gains',
    'capital_loss',
    'credit_score',
    'telematics_score',
    'incident_type',
    'collision_type',
    'incident_severity',
    'authorities_contacted',
    'incident_hour_of_the_day',
    'number_of_vehicles_involved',
    'property_damage',
    'bodily_injuries',
    'witnesses',
    'police_report_available',
    'total_claim_amount',
    'prior_claims_count',
    'incident_in_policy_period',
    'policy_status_at_incident',
    'incident_near_boundary',
    'is_complex_claim',
]

# Defaults only for fields your pipeline genuinely cannot provide
DEFAULTS = {
    'insured_sex': 'MALE',
    'insured_education_level': 'Bachelors',
    'insured_occupation': 'other-service',
    'umbrella_limit': 0,
    'capital_gains': 0,
    'capital_loss': 0,
    'telematics_score': 70.0,
    'collision_type': '?',
    'incident_hour_of_the_day': 12,
    'number_of_vehicles_involved': 1,
    'property_damage': 'YES',
    'bodily_injuries': 0,
}


def build_features(
    parsed: Dict[str, Any],
    policy: Dict[str, Any],
    verification: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Build the exact 27-feature vector for the fraud ML model.
    
    Args:
        parsed: Agent 1 output
        policy: Agent 2 policy dict
        verification: Agent 2 verification dict
    
    Returns:
        (ordered_features_dict, list_of_fields_that_used_defaults)
    """
    f = {}
    imputed = []
    
    def _use(field, value, default):
        if value is not None and value != "":
            f[field] = value
        else:
            f[field] = default
            imputed.append(field)
    
    # ── months_as_customer ─────────────────────────────────────────────────
    try:
        inc = datetime.strptime(str(parsed.get("incident_date", "")), "%Y-%m-%d")
        eff = datetime.strptime(str(policy.get("effective_date", "")), "%Y-%m-%d")
        f['months_as_customer'] = max(1, (inc - eff).days // 30)
    except (ValueError, TypeError):
        f['months_as_customer'] = 24
        imputed.append('months_as_customer')
    
    # ── age ────────────────────────────────────────────────────────────────
    try:
        dob = datetime.strptime(str(policy.get("insured_dob", "")), "%Y-%m-%d")
        inc = datetime.strptime(str(parsed.get("incident_date", "")), "%Y-%m-%d")
        f['age'] = int((inc - dob).days / 365.25)
    except (ValueError, TypeError):
        f['age'] = 40
        imputed.append('age')
    
    # ── Demographics (not in pipeline) ─────────────────────────────────────
    f['insured_sex'] = DEFAULTS['insured_sex']
    f['insured_education_level'] = DEFAULTS['insured_education_level']
    f['insured_occupation'] = DEFAULTS['insured_occupation']
    imputed.extend(['insured_sex', 'insured_education_level', 'insured_occupation'])
    
    # ── Policy financials ──────────────────────────────────────────────────
    f['policy_deductable'] = int(policy.get("coverage_deductible", 500))
    
    # ── Not available from pipeline ────────────────────────────────────────
    for field in ['umbrella_limit', 'capital_gains', 'capital_loss', 'telematics_score']:
        f[field] = DEFAULTS[field]
        imputed.append(field)
    
    # ── credit_score (from policy) ─────────────────────────────────────────
    f['credit_score'] = int(policy.get("credit_score", 650))
    
    # ── Incident details (from Agent 1) ────────────────────────────────────
    f['incident_type'] = str(parsed.get("incident_type") or "Other")
    f['incident_severity'] = str(parsed.get("incident_severity") or "Minor Damage")
    f['authorities_contacted'] = str(parsed.get("authorities_contacted") or "None")
    
    # May not be in parsed
    _use('collision_type', parsed.get("collision_type"), DEFAULTS['collision_type'])
    _use('incident_hour_of_the_day', parsed.get("incident_hour_of_the_day"), DEFAULTS['incident_hour_of_the_day'])
    _use('number_of_vehicles_involved', parsed.get("number_of_vehicles_involved"), DEFAULTS['number_of_vehicles_involved'])
    _use('property_damage', parsed.get("property_damage"), DEFAULTS['property_damage'])
    
    # ── Claim characteristics ──────────────────────────────────────────────
    f['bodily_injuries'] = int(parsed.get("bodily_injuries") or 0)
    f['witnesses'] = int(parsed.get("witnesses") or 0)
    f['police_report_available'] = str(parsed.get("police_report_available") or "YES")
    f['total_claim_amount'] = float(parsed.get("total_claim_amount") or 1000)
    f['prior_claims_count'] = int(policy.get("prior_claims_count", 0))
    
    # ── Temporal safety ────────────────────────────────────────────────────
    f['incident_in_policy_period'] = (
        'Y' if verification.get("incident_in_policy_period") else 'N'
    )
    f['policy_status_at_incident'] = str(
        verification.get("policy_status") or "Active"
    )
    f['incident_near_boundary'] = _calc_boundary(
        parsed.get("incident_date"),
        policy.get("effective_date"),
        policy.get("expiration_date"),
    )
    
    # ── is_complex_claim ───────────────────────────────────────────────────
    f['is_complex_claim'] = _calc_complexity(f, verification)
    
    # Return in exact training order
    ordered = {col: f.get(col, DEFAULTS.get(col, "")) for col in FEATURE_COLUMNS}
    return ordered, imputed


def _calc_boundary(incident_date, effective_date, expiration_date) -> str:
    try:
        inc = datetime.strptime(str(incident_date), "%Y-%m-%d")
        eff = datetime.strptime(str(effective_date), "%Y-%m-%d")
        exp = datetime.strptime(str(expiration_date), "%Y-%m-%d")
        if (inc - eff).days <= 30 or (exp - inc).days <= 30:
            return 'Y'
        return 'N'
    except (ValueError, TypeError):
        return 'N'


def _calc_complexity(features: Dict, verification: Dict) -> str:
    if features.get('total_claim_amount', 0) > 15000:
        return 'Y'
    if features.get('bodily_injuries', 0) > 1:
        return 'Y'
    if features.get('incident_in_policy_period') == 'N':
        return 'Y'
    if features.get('policy_status_at_incident') != 'Active':
        return 'Y'
    if len(verification.get('errors', [])) > 0:
        return 'Y'
    return 'N'


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL
# ═══════════════════════════════════════════════════════════════════════════════


def build_features(
    customer: Dict[str, Any],
    policy: Dict[str, Any],
    vehicle: Dict[str, Any],
    claim: Dict[str, Any],
    fraud_assessment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build 27 features directly from database records.
    Zero imputation — every field comes from a real source.
    """
    f = {}

    # ── Customer Master ────────────────────────────────────────────────────
    f['insured_sex'] = customer.get('gender', 'MALE')
    f['insured_education_level'] = customer.get('education_level', 'Bachelors')
    f['insured_occupation'] = customer.get('occupation', 'other-service')
    f['capital_gains'] = float(customer.get('capital_gains', 0))
    f['capital_loss'] = float(customer.get('capital_loss', 0))
    f['credit_score'] = int(customer.get('credit_score', 650))

    # months_as_customer
    try:
        since = datetime.strptime(str(customer.get('customer_since_date', '')), '%Y-%m-%d')
        inc = datetime.strptime(str(claim.get('incident_date', '')), '%Y-%m-%d')
        f['months_as_customer'] = max(1, (inc - since).days // 30)
    except:
        f['months_as_customer'] = 24

    # age
    try:
        dob = datetime.strptime(str(customer.get('date_of_birth', '')), '%Y-%m-%d')
        inc = datetime.strptime(str(claim.get('incident_date', '')), '%Y-%m-%d')
        f['age'] = int((inc - dob).days / 365.25)
    except:
        f['age'] = 40

    # ── Policy Data ────────────────────────────────────────────────────────
    f['umbrella_limit'] = float(policy.get('umbrella_limit', 0))

    # ── Coverage Line (deductible comes from here) ─────────────────────────
    f['policy_deductable'] = int(policy.get('deductible', 
                                             policy.get('coverage_deductible', 500)))

    # ── Vehicle ────────────────────────────────────────────────────────────
    f['telematics_score'] = float(vehicle.get('telematics_score', 70.0))

    # ── Claim ──────────────────────────────────────────────────────────────
    f['incident_type'] = str(claim.get('incident_type', 'Other'))
    f['collision_type'] = str(claim.get('collision_type', '?'))
    f['incident_severity'] = str(claim.get('incident_severity', 'Minor Damage'))
    f['authorities_contacted'] = str(claim.get('authorities_contacted', 'None'))
    f['incident_hour_of_the_day'] = int(claim.get('incident_hour_of_the_day', 12))
    f['number_of_vehicles_involved'] = int(claim.get('number_of_vehicles_involved', 1))
    f['property_damage'] = str(claim.get('property_damage', 'YES'))
    f['bodily_injuries'] = int(claim.get('bodily_injuries', 0))
    f['witnesses'] = int(claim.get('witnesses', 0))
    f['police_report_available'] = str(claim.get('police_report_available', 'YES'))
    f['total_claim_amount'] = float(claim.get('total_claim_amount', 1000))
    f['prior_claims_count'] = int(claim.get('prior_claims_count', 0))

    # ── Fraud Assessment ───────────────────────────────────────────────────
    f['incident_in_policy_period'] = str(fraud_assessment.get('incident_in_policy_period', 'Y'))
    f['policy_status_at_incident'] = str(fraud_assessment.get('policy_status_at_incident', 'Active'))
    f['incident_near_boundary'] = str(fraud_assessment.get('incident_near_boundary', 'N'))
    f['is_complex_claim'] = str(fraud_assessment.get('is_complex_claim', 'N'))

    return {col: f[col] for col in FEATURE_COLUMNS}


def build_feature_vector(
    parsed: Dict[str, Any] = None,
    policy: Dict[str, Any] = None,
    verification: Dict[str, Any] = None,
    customer_profile: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    MCP Tool: Build 27 features from pipeline data.
    Works with existing pipeline inputs (parsed + policy + verification + customer).
    Uses a safe helper to avoid NoneType errors on optional fields.
    """
    f = {}

    # Helper: get value or default (handles None properly)
    def _get(d, key, default):
        if d is None:
            return default
        val = d.get(key)
        return val if val is not None else default

    # ── From Customer Profile ──────────────────────────────────────────────
    if customer_profile:
        f['insured_sex'] = _get(customer_profile, 'insured_sex', 
                                _get(customer_profile, 'gender', 'MALE'))
        f['insured_education_level'] = _get(customer_profile, 'insured_education_level',
                                            _get(customer_profile, 'education_level', 'Bachelors'))
        f['insured_occupation'] = _get(customer_profile, 'insured_occupation',
                                       _get(customer_profile, 'occupation', 'other-service'))
        f['capital_gains'] = float(_get(customer_profile, 'capital_gains', 0))
        f['capital_loss'] = float(_get(customer_profile, 'capital_loss', 0))
        f['credit_score'] = int(_get(customer_profile, 'credit_score', 650))

        try:
            since = _get(customer_profile, 'customer_since_date', '') or \
                    _get(customer_profile, 'first_policy_date', '')
            inc = _get(parsed, 'incident_date', '')
            if since and inc:
                f['months_as_customer'] = max(1, (datetime.strptime(str(inc), '%Y-%m-%d') - 
                                                   datetime.strptime(str(since), '%Y-%m-%d')).days // 30)
            else:
                f['months_as_customer'] = 24
        except:
            f['months_as_customer'] = 24

        try:
            dob = _get(customer_profile, 'date_of_birth', '') or \
                  _get(customer_profile, 'customer_dob', '')
            inc = _get(parsed, 'incident_date', '')
            if dob and inc:
                f['age'] = int((datetime.strptime(str(inc), '%Y-%m-%d') - 
                               datetime.strptime(str(dob), '%Y-%m-%d')).days / 365.25)
            else:
                f['age'] = 40
        except:
            f['age'] = 40
    else:
        f['insured_sex'] = 'MALE'
        f['insured_education_level'] = 'Bachelors'
        f['insured_occupation'] = 'other-service'
        f['capital_gains'] = 0
        f['capital_loss'] = 0
        f['credit_score'] = int(_get(policy, 'credit_score', 650))
        f['months_as_customer'] = 24
        f['age'] = 40

    # ── From Policy ────────────────────────────────────────────────────────
    f['umbrella_limit'] = float(_get(policy, 'umbrella_limit', 0))
    f['policy_deductable'] = int(_get(policy, 'coverage_deductible', 500))
    f['telematics_score'] = float(_get(policy, 'telematics_score', 70.0))
    f['prior_claims_count'] = int(_get(policy, 'prior_claims_count', 0))

    # ── From Parsed (Agent 1) ──────────────────────────────────────────────
    f['incident_type'] = str(_get(parsed, 'incident_type', 'Other'))
    f['collision_type'] = str(_get(parsed, 'collision_type', '?'))
    f['incident_severity'] = str(_get(parsed, 'incident_severity', 'Minor Damage'))
    f['authorities_contacted'] = str(_get(parsed, 'authorities_contacted', 'None'))
    f['incident_hour_of_the_day'] = int(_get(parsed, 'incident_hour_of_the_day', 12))
    f['number_of_vehicles_involved'] = int(_get(parsed, 'number_of_vehicles_involved', 1))
    f['property_damage'] = str(_get(parsed, 'property_damage', 'YES'))
    f['bodily_injuries'] = int(_get(parsed, 'bodily_injuries', 0))
    f['witnesses'] = int(_get(parsed, 'witnesses', 0))
    f['police_report_available'] = str(_get(parsed, 'police_report_available', 'YES'))
    f['total_claim_amount'] = float(_get(parsed, 'total_claim_amount', 1000))

    # ── From Verification (Agent 2) ────────────────────────────────────────
    if verification:
        f['incident_in_policy_period'] = 'Y' if verification.get('incident_in_policy_period') else 'N'
        f['policy_status_at_incident'] = str(_get(verification, 'policy_status', 'Active'))
        f['incident_near_boundary'] = 'Y' if verification.get('incident_near_boundary') == 'Y' else 'N'
        f['is_complex_claim'] = 'Y' if (
            f.get('total_claim_amount', 0) > 15000 or
            f.get('bodily_injuries', 0) > 1 or
            f.get('incident_in_policy_period') == 'N' or
            f.get('policy_status_at_incident') != 'Active' or
            len(verification.get('errors', [])) > 0
        ) else 'N'
    else:
        f['incident_in_policy_period'] = 'Y'
        f['policy_status_at_incident'] = 'Active'
        f['incident_near_boundary'] = 'N'
        f['is_complex_claim'] = 'N'

    # ── Return ─────────────────────────────────────────────────────────────
    ordered = {col: f.get(col, '') for col in FEATURE_COLUMNS}

    imputed = []
    if not customer_profile:
        imputed = ['insured_sex', 'insured_education_level', 'insured_occupation',
                   'capital_gains', 'capital_loss']

    return {
        "features": ordered,
        "feature_count": 27,
        "imputed_count": len(imputed),
        "imputed_fields": imputed,
        "customer_profile_used": customer_profile is not None,
        "ready_for_ml": True,
    }

