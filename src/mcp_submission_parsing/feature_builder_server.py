"""
Agent 4 — Feature Builder
Builds the exact 39-column feature vector matching the trained fraud ML model.
"""

from datetime import datetime
from typing import Dict, Any, List


# Exact 39 features your model expects — order must match training
FEATURE_COLUMNS = [
    'months_as_customer',
    'age',
    'insured_sex',
    'insured_education_level',
    'insured_occupation',
    'insured_hobbies',
    'insured_relationship',
    'policy_state',
    'policy_csl',
    'policy_deductable',
    'policy_annual_premium',
    'umbrella_limit',
    'capital_gains',
    'capital_loss',
    'credit_score',
    'telematics_score',
    'incident_type',
    'collision_type',
    'incident_severity',
    'authorities_contacted',
    'incident_state',
    'incident_location',
    'incident_hour_of_the_day',
    'number_of_vehicles_involved',
    'property_damage',
    'bodily_injuries',
    'witnesses',
    'police_report_available',
    'total_claim_amount',
    'prior_claims_count',
    'auto_make',
    'auto_year',
    'injury_claim',
    'property_claim',
    'vehicle_claim',
    'incident_in_policy_period',
    'policy_status_at_incident',
    'incident_near_boundary',
    'is_complex_claim',
]

# Safe defaults for missing fields (should rarely be used after data alignment)
DEFAULTS = {
    'months_as_customer': 24,
    'age': 40,
    'insured_sex': 'MALE',
    'insured_education_level': 'Bachelors',
    'insured_occupation': 'other-service',
    'insured_hobbies': 'reading',
    'insured_relationship': 'not-in-family',
    'policy_state': 'CA',
    'policy_csl': '100/300',
    'policy_deductable': 500,
    'policy_annual_premium': 1000.0,
    'umbrella_limit': 0,
    'capital_gains': 0,
    'capital_loss': 0,
    'credit_score': 650,
    'telematics_score': 70.0,
    'incident_type': 'Other',
    'collision_type': '?',
    'incident_severity': 'Minor Damage',
    'authorities_contacted': 'None',
    'incident_state': 'CA',
    'incident_location': 'Highway',
    'incident_hour_of_the_day': 12,
    'number_of_vehicles_involved': 1,
    'property_damage': 'YES',
    'bodily_injuries': 0,
    'witnesses': 0,
    'police_report_available': 'YES',
    'total_claim_amount': 1000.0,
    'prior_claims_count': 0,
    'auto_make': 'Unknown',
    'auto_year': 2015,
    'injury_claim': 0,
    'property_claim': 0,
    'vehicle_claim': 0,
    'incident_in_policy_period': 'Y',
    'policy_status_at_incident': 'Active',
    'incident_near_boundary': 'N',
    'is_complex_claim': 'N',
}


def build_feature_vector(
    parsed: Dict[str, Any] = None,
    policy: Dict[str, Any] = None,
    verification: Dict[str, Any] = None,
    customer_profile: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Build the exact 39-feature vector for the fraud ML model.

    Args:
        parsed: Output from Agent 1 (submission parser)
        policy: Output from Agent 2 (policy lookup) – as a PolicyInfo dict
        verification: Output from Agent 2 (verification results)
        customer_profile: Optional Customer_Master data

    Returns:
        Dictionary with 'features', 'imputed_fields', and 'ready_for_ml' flag.
    """
    f = {}
    imputed_fields = []

    def _get(d, key, default):
        if d is None:
            return default
        val = d.get(key)
        return val if val is not None and val != "" else default

    # Initialize with safe defaults
    for col in FEATURE_COLUMNS:
        f[col] = DEFAULTS.get(col, '')

    # ── Customer Profile overrides ──────────────────────────────────────
    if customer_profile:
        f['insured_sex'] = _get(customer_profile, 'insured_sex',
                                _get(customer_profile, 'gender', DEFAULTS['insured_sex']))
        f['insured_education_level'] = _get(customer_profile, 'insured_education_level',
                                            _get(customer_profile, 'education_level',
                                                 DEFAULTS['insured_education_level']))
        f['insured_occupation'] = _get(customer_profile, 'insured_occupation',
                                       _get(customer_profile, 'occupation',
                                            DEFAULTS['insured_occupation']))
        f['insured_hobbies'] = _get(customer_profile, 'insured_hobbies', DEFAULTS['insured_hobbies'])
        f['insured_relationship'] = _get(customer_profile, 'insured_relationship',
                                         DEFAULTS['insured_relationship'])
        f['credit_score'] = int(_get(customer_profile, 'credit_score', DEFAULTS['credit_score']))
        f['capital_gains'] = float(_get(customer_profile, 'capital_gains', DEFAULTS['capital_gains']))
        f['capital_loss'] = float(_get(customer_profile, 'capital_loss', DEFAULTS['capital_loss']))

        # months_as_customer from customer_since_date
        try:
            since = _get(customer_profile, 'customer_since_date', '') or \
                    _get(customer_profile, 'first_policy_date', '')
            inc = _get(parsed, 'incident_date', '')
            if since and inc:
                f['months_as_customer'] = max(1, (datetime.strptime(str(inc), '%Y-%m-%d') -
                                                  datetime.strptime(str(since), '%Y-%m-%d')).days // 30)
            else:
                imputed_fields.append('months_as_customer')
        except Exception:
            imputed_fields.append('months_as_customer')

        # age from date_of_birth
        try:
            dob = _get(customer_profile, 'date_of_birth', '') or \
                  _get(customer_profile, 'customer_dob', '')
            inc = _get(parsed, 'incident_date', '')
            if dob and inc:
                f['age'] = int((datetime.strptime(str(inc), '%Y-%m-%d') -
                                datetime.strptime(str(dob), '%Y-%m-%d')).days / 365.25)
            else:
                imputed_fields.append('age')
        except Exception:
            imputed_fields.append('age')
    else:
        imputed_fields.extend(['insured_sex', 'insured_education_level', 'insured_occupation',
                               'insured_hobbies', 'insured_relationship', 'credit_score',
                               'capital_gains', 'capital_loss', 'months_as_customer', 'age'])

    # ── Policy Data overrides ───────────────────────────────────────────
    if policy:
        f['policy_state'] = _get(policy, 'insured_state', DEFAULTS['policy_state'])
        f['policy_csl'] = _get(policy, 'policy_csl', DEFAULTS['policy_csl'])
        f['policy_deductable'] = int(_get(policy, 'coverage_deductible', DEFAULTS['policy_deductable']))
        f['policy_annual_premium'] = float(_get(policy, 'total_annual_premium', DEFAULTS['policy_annual_premium']))
        f['umbrella_limit'] = float(_get(policy, 'umbrella_limit', DEFAULTS['umbrella_limit']))
        f['telematics_score'] = float(_get(policy, 'telematics_score', DEFAULTS['telematics_score']))
        f['prior_claims_count'] = int(_get(policy, 'prior_claims_count', DEFAULTS['prior_claims_count']))
        f['auto_make'] = _get(policy, 'vehicle_make', DEFAULTS['auto_make'])
        f['auto_year'] = int(_get(policy, 'vehicle_year', DEFAULTS['auto_year']))

        # Fallback credit score if not in customer profile
        if not customer_profile:
            f['credit_score'] = int(_get(policy, 'credit_score', DEFAULTS['credit_score']))

    # ── Parsed Submission overrides ─────────────────────────────────────
    if parsed:
        f['incident_type'] = str(_get(parsed, 'incident_type', DEFAULTS['incident_type']))
        f['collision_type'] = str(_get(parsed, 'collision_type', DEFAULTS['collision_type']))
        f['incident_severity'] = str(_get(parsed, 'incident_severity', DEFAULTS['incident_severity']))
        f['authorities_contacted'] = str(_get(parsed, 'authorities_contacted', DEFAULTS['authorities_contacted']))
        f['incident_state'] = str(_get(parsed, 'incident_state', DEFAULTS['incident_state']))
        f['incident_location'] = str(_get(parsed, 'incident_location', DEFAULTS['incident_location']))
        f['incident_hour_of_the_day'] = int(_get(parsed, 'incident_hour_of_the_day',
                                                 DEFAULTS['incident_hour_of_the_day']))
        f['number_of_vehicles_involved'] = int(_get(parsed, 'number_of_vehicles_involved',
                                                    DEFAULTS['number_of_vehicles_involved']))
        f['property_damage'] = str(_get(parsed, 'property_damage', DEFAULTS['property_damage']))
        f['bodily_injuries'] = int(_get(parsed, 'bodily_injuries', DEFAULTS['bodily_injuries']))
        f['witnesses'] = int(_get(parsed, 'witnesses', DEFAULTS['witnesses']))
        f['police_report_available'] = str(_get(parsed, 'police_report_available',
                                                DEFAULTS['police_report_available']))
        f['total_claim_amount'] = float(_get(parsed, 'total_claim_amount', DEFAULTS['total_claim_amount']))

        # Claim breakdown: set vehicle_claim = total_claim_amount (simplified),
        # injury and property to 0. This ensures calculated_total == total_claim_amount.
        # For better accuracy, you could parse these from the claim text.
        f['vehicle_claim'] = f['total_claim_amount']
        f['injury_claim'] = 0.0
        f['property_claim'] = 0.0

    # ── Verification overrides ──────────────────────────────────────────
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

    # Return features in the exact order
    ordered_features = {col: f.get(col, DEFAULTS.get(col, '')) for col in FEATURE_COLUMNS}

    return {
        "features": ordered_features,
        "feature_count": len(FEATURE_COLUMNS),
        "imputed_count": len(imputed_fields),
        "imputed_fields": imputed_fields,
        "customer_profile_used": customer_profile is not None,
        "ready_for_ml": True,
    }


def get_feature_columns() -> List[str]:
    """Return the list of feature columns in the exact order expected by the model."""
    return FEATURE_COLUMNS.copy()


def get_default_values() -> Dict[str, Any]:
    """Return the default values dictionary for all features."""
    return DEFAULTS.copy()