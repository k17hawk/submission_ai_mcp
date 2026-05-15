# test_pipeline.py
"""
Serial integration test: Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5
Picks a random active policy, generates a compatible claim,
runs the full pipeline (including fraud ML prediction),
and saves the generated features + prediction to CSV.
"""

import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_submission_parsing.parser_server import SubmissionParser
from src.mcp_submission_parsing.policy_lookup_server import (
    PolicyLookupService,
    lookup_policy,
    verify_identity,
    check_coverage,
    pre_fill_claim,
    INCIDENT_TO_COVERAGE,
)
from src.mcp_submission_parsing.rule_checker_server import check_risk_rules
from src.mcp_submission_parsing.feature_builder_server import build_feature_vector
from src.mcp_submission_parsing.fraud_detection_server import predict_fraud

import pandas as pd

# ── Setup ────────────────────────────────────────────────────
parser = SubmissionParser()
DATA_PATH = "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
service = PolicyLookupService(DATA_PATH)

# Extend coverage mapping so "Other" works with all coverage types
INCIDENT_TO_COVERAGE.update({
    "Other": ["COMP", "COLL", "LIAB", "UNINSMOT", "MED", "PIP"],
})

# Realistic incident types and severities (based on data generator)
INCIDENT_TYPES = [
    "Single Vehicle Collision",
    "Multi-vehicle Collision",
    "Parked Car",
    "Vehicle Theft",
    "Other",
]
# Weights reflect typical claim volumes
INCIDENT_WEIGHTS = [0.35, 0.30, 0.15, 0.10, 0.10]

SEVERITIES = ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"]
# Realistic severity distribution
SEVERITY_WEIGHTS = [0.45, 0.30, 0.15, 0.10]


def pick_active_policy(service: PolicyLookupService):
    """Pick a random active policy with valid date range of at least 30 days."""
    active = [p for p in service._policies.values() if p.policy_status == "Active"]
    if not active:
        return None

    for _ in range(20):
        policy = random.choice(active)
        try:
            eff = datetime.strptime(policy.effective_date, "%Y-%m-%d")
            exp = datetime.strptime(policy.expiration_date, "%Y-%m-%d")
            if eff < exp and (exp - eff).days >= 30:
                return policy
        except (ValueError, TypeError):
            continue

    return None


def split_claim_amount(total: float, incident_type: str, severity: str) -> tuple:
    """
    Realistic split of total claim amount into vehicle, injury, property.
    Based on incident type and severity (matches generator logic).
    """
    if incident_type == "Vehicle Theft":
        return (total, 0.0, 0.0)
    if severity == "Total Loss":
        return (total * 0.9, total * 0.05, total * 0.05)
    elif severity == "Major Damage":
        return (total * 0.7, total * 0.2, total * 0.1)
    elif severity == "Minor Damage":
        return (total * 0.6, total * 0.3, total * 0.1)
    else:  # Trivial Damage
        return (total * 0.8, total * 0.1, total * 0.1)


def generate_claim_amount(severity: str, vehicle_value: float) -> float:
    """Generate a plausible claim amount based on severity and vehicle value."""
    if severity == "Total Loss":
        return round(vehicle_value * random.uniform(0.85, 1.10), 2)
    elif severity == "Major Damage":
        return round(vehicle_value * random.uniform(0.25, 0.65), 2)
    elif severity == "Minor Damage":
        return round(vehicle_value * random.uniform(0.05, 0.25), 2)
    else:  # Trivial Damage
        return round(random.uniform(200, 1500), 2)


# ═══════════════════════════════════════════════════════════════
# 1. PICK POLICY & GENERATE INCIDENT
# ═══════════════════════════════════════════════════════════════

policy = pick_active_policy(service)
if not policy:
    print("No active policies found with valid date ranges.")
    sys.exit(1)

eff = datetime.strptime(policy.effective_date, "%Y-%m-%d")
exp = datetime.strptime(policy.expiration_date, "%Y-%m-%d")

# Generate incident at least 31 days after policy start
min_days = min(31, max(0, (exp - eff).days))
delta = random.randint(min_days, max(min_days, (exp - eff).days))
inc_date = eff + timedelta(days=delta)
inc_date_str = inc_date.strftime("%Y-%m-%d")

# Pick compatible incident type (respect coverage mapping)
compatible_types = [
    inc for inc in INCIDENT_TYPES
    if policy.coverage_code in INCIDENT_TO_COVERAGE.get(inc, [])
]
if not compatible_types:
    print(f"❌ Policy {policy.policy_number} has coverage '{policy.coverage_code}'")
    print(f"   No incident types map to this coverage.")
    sys.exit(1)

# Select incident type with realistic distribution (but limited to compatible)
# Filter weights to only compatible types
filtered_weights = [INCIDENT_WEIGHTS[INCIDENT_TYPES.index(t)] for t in compatible_types]
inc_type = random.choices(compatible_types, weights=filtered_weights, k=1)[0]

# Select severity with realistic distribution
severity = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS, k=1)[0]

# Generate claim amount and split
claim_amount = generate_claim_amount(severity, policy.vehicle_value)
vehicle_claim, injury_claim, property_claim = split_claim_amount(claim_amount, inc_type, severity)

# Build claim text (includes the incident type and severity)
claim_text = (
    f"Hi, I need to file a claim. On {inc_date_str} I was involved in a "
    f"{inc_type} on Highway in Springfield, IL. "
    f"My vehicle, a {policy.vehicle_year} {policy.vehicle_make} {policy.vehicle_model}, "
    f"sustained {severity}. The estimated damage is about ${claim_amount:,.2f}. "
    f"Policy number is {policy.policy_number}. "
    f"My name is {policy.insured_name}. Authorities contacted: Police."
)


# ═══════════════════════════════════════════════════════════════
# 2. AGENT 1: SUBMISSION PARSER
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("AGENT 1: Submission Parser")
print("=" * 60)

result = parser.parse(claim_text, "email")
parsed = result.__dict__

print(f"Policy: {policy.policy_number} | {policy.policy_status}")
print(f"Period: {policy.effective_date} → {policy.expiration_date}")
print(f"Vehicle: {policy.vehicle_year} {policy.vehicle_make} {policy.vehicle_model} "
      f"(${policy.vehicle_value:,.2f})")
print(f"Coverage: {policy.coverage_code} ({policy.coverage_name})")
print(f"Limit: ${policy.coverage_limit:,.2f} | Deductible: ${policy.coverage_deductible:,.2f}")
print(f"Insured: {policy.insured_name}")
print()

print(f"Incident Date: {inc_date_str} (day {delta} of policy)")
print(f"Type: {inc_type} | Severity: {severity}")
print(f"Amount: ${claim_amount:,.2f}")
print(f"Claim Breakdown: Vehicle=${vehicle_claim:,.2f}, Injury=${injury_claim:,.2f}, Property=${property_claim:,.2f}")
print()

print("Parsed fields:")
for k, v in parsed.items():
    if v is not None:
        print(f"  {k}: {v}")
print(f"  complete: {result.is_complete()}")
print(f"  confidence: {result.extraction_confidence:.1%}")
print()

# Inject missing fields (including claim breakdown)
if not parsed.get("total_claim_amount"):
    parsed["total_claim_amount"] = claim_amount
if not parsed.get("police_report_available"):
    parsed["police_report_available"] = "YES"
if not parsed.get("witnesses"):
    parsed["witnesses"] = 1
if not parsed.get("bodily_injuries"):
    parsed["bodily_injuries"] = 0
# Inject claim breakdown (parser doesn't extract these from text yet)
parsed["vehicle_claim"] = vehicle_claim
parsed["injury_claim"] = injury_claim
parsed["property_claim"] = property_claim

print(f"  ℹ️  Injected: total_claim_amount=${claim_amount:,.2f}, "
      f"police_report=YES, witnesses=1, bodily_injuries=0, "
      f"vehicle_claim=${vehicle_claim:,.2f}, injury_claim=${injury_claim:,.2f}, "
      f"property_claim=${property_claim:,.2f}")
print()


# ═══════════════════════════════════════════════════════════════
# 3. AGENT 2: IDENTITY & POLICY LOOKUP
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("AGENT 2: Identity & Policy Lookup")
print("=" * 60)

# 3a. Lookup policy
print("\n>>> lookup_policy")
presp = lookup_policy(parsed["policy_number"])
print(f"Found: {presp['found']}")
if not presp["found"]:
    print("❌ Policy not found – aborting.")
    sys.exit(1)

policy_data = presp.get("policy", {})

# 3b. Verify identity
print("\n>>> verify_identity (by name)")
iresp = verify_identity(
    customer_name=policy.insured_name,
    policy_number=parsed["policy_number"]
)
print(f"Verified: {iresp['verified']} — {iresp['message']}")

customer_id = iresp.get("customer_id") if iresp["verified"] else policy_data.get("customer_id")

# 3c. Check coverage
print("\n>>> check_coverage")
cov = check_coverage(
    parsed["policy_number"],
    parsed["incident_type"],
    parsed["incident_date"],
    claim_amount=parsed.get("total_claim_amount", 0) or 0,
)
print(f"Covered: {cov['covered']}")
if not cov['covered']:
    print(f"  Reason: {cov['reason']}")
print(f"  Deductible: ${cov.get('deductible', 0):,.2f}")

# 3d. Pre-fill claim
print("\n>>> pre_fill_claim")
claim = pre_fill_claim(parsed)
verification = claim.get('verification', {})
print(f"Status: {claim.get('status')}")
errors = verification.get('errors', [])
warnings_list = verification.get('warnings', [])
if errors:
    print(f"Errors: {errors}")
if warnings_list:
    print(f"Warnings: {warnings_list}")


# ═══════════════════════════════════════════════════════════════
# 4. AGENT 3: RISK RULE CHECKER
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("AGENT 3: Risk Rule Checker")
print("=" * 60)

risk = check_risk_rules(parsed, verification)

print(f"\nPassed: {risk['passed']}")
print(f"Risk Score: {risk['risk_score']} ({risk['risk_level']})")
print(f"Auto Decision: {risk['auto_decision']}")
print(f"Needs SIU: {risk['requires_siu']}")
print(f"Needs Adjuster: {risk['requires_adjuster']}")

if risk.get('violations'):
    print(f"\n❌ Violations ({len(risk['violations'])}):")
    for v in risk['violations']:
        print(f"  [{v['severity']}] {v['rule_id']} {v['rule_name']}")
        print(f"    → {v['message']}")

if risk.get('warnings'):
    print(f"\n⚠️  Warnings ({len(risk['warnings'])}):")
    for w in risk['warnings']:
        print(f"  [{w['severity']}] {w['rule_name']}")
        print(f"    → {w['message']}")

if not risk.get('violations') and not risk.get('warnings'):
    print("\n✅ Clean claim — no violations or warnings!")


# ═══════════════════════════════════════════════════════════════
# 5. AGENT 4: FEATURE BUILDER (for Fraud ML)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("AGENT 4: Feature Builder (Fraud ML Input)")
print("=" * 60)

# Fetch customer profile
customer_profile = None
if customer_id:
    customer_data = service.get_customer(customer_id)
    if customer_data:
        customer_profile = customer_data
        print(f"\nCustomer profile loaded: {customer_id}")
        print(f"  Name: {customer_profile.get('customer_name')}")
        print(f"  DOB: {customer_profile.get('customer_dob')}")
        print(f"  Segment: {customer_profile.get('customer_segment')}")
        print(f"  Lifetime claims: {customer_profile.get('lifetime_claims')}")
        print(f"  Credit score: {customer_profile.get('credit_score')}")
        print(f"  Sex: {customer_profile.get('insured_sex', 'N/A')}")
        print(f"  Education: {customer_profile.get('insured_education_level', 'N/A')}")
        print(f"  Occupation: {customer_profile.get('insured_occupation', 'N/A')}")
        print(f"  Hobbies: {customer_profile.get('insured_hobbies', 'N/A')}")
        print(f"  Relationship: {customer_profile.get('insured_relationship', 'N/A')}")
    else:
        print(f"\n⚠️  Customer {customer_id} not found in Customer_Master")

# Build feature vector (now includes prior_claims_count from policy data)
feature_result = build_feature_vector(parsed, policy_data, verification, customer_profile)

print(f"\nFeatures: {feature_result['feature_count']}/39")   # Note: 39, not 27
print(f"Customer profile used: {feature_result['customer_profile_used']}")
print(f"Imputed count: {feature_result['imputed_count']}")
if feature_result['imputed_fields']:
    print(f"Imputed fields: {feature_result['imputed_fields']}")
print(f"Ready for ML: {feature_result['ready_for_ml']}")

features = feature_result['features']


# ═══════════════════════════════════════════════════════════════
# 6. AGENT 5: FRAUD DETECTION ML
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("AGENT 5: Fraud Detection ML")
print("=" * 60)

try:
    fraud_result = predict_fraud(features)
    print(f"\nFraud Probability: {fraud_result['fraud_probability']:.4f}")
    print(f"Fraud Flag:        {fraud_result['fraud_flag']}")
    print(f"Risk Level:        {fraud_result['risk_level']}")
    print(f"Threshold Used:    {fraud_result['threshold_used']:.4f}")
    print(f"Requires SIU:      {fraud_result['requires_siu']}")
except Exception as e:
    print(f"\n❌ Fraud detection failed: {e}")
    fraud_result = {
        'fraud_probability': None,
        'fraud_flag': 'ERROR',
        'risk_level': 'UNKNOWN',
        'threshold_used': None,
        'requires_siu': False,
    }


# ═══════════════════════════════════════════════════════════════
# 7. PIPELINE SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"  Policy: {policy.policy_number} ({policy.coverage_code} → {policy.coverage_name})")
print(f"  Insured: {policy.insured_name}")
print(f"  Vehicle: {policy.vehicle_year} {policy.vehicle_make} {policy.vehicle_model} "
      f"(${policy.vehicle_value:,.2f})")
print(f"  Incident: {inc_date_str} | {inc_type} | {severity}")
print(f"  Claim Amount: ${claim_amount:,.2f} "
      f"(deductible: ${policy.coverage_deductible:,.2f}, "
      f"limit: ${policy.coverage_limit:,.2f})")
print(f"  Claim Breakdown: Vehicle=${vehicle_claim:,.2f}, Injury=${injury_claim:,.2f}, Property=${property_claim:,.2f}")
print(f"  ─────────────────────────────────────")
print(f"  Agent 1 — Parsing: {result.extraction_confidence:.1%} confidence, "
      f"{'complete' if result.is_complete() else 'incomplete'}")
print(f"  Agent 2 — Coverage: {'PASS' if cov.get('covered') else 'FAIL'}")
print(f"  Agent 3 — Risk: {risk['risk_score']} ({risk['risk_level']}) → "
      f"{risk['auto_decision'] or 'MANUAL REVIEW'}")
print(f"  Agent 4 — Features: {feature_result['feature_count']} columns ready for ML")
print(f"  Agent 5 — Fraud ML: {fraud_result['fraud_flag']} "
      f"(prob={fraud_result['fraud_probability']:.4f}, "
      f"risk={fraud_result['risk_level']})")
print(f"  ─────────────────────────────────────")
print(f"  SIU Required: {'Yes' if risk['requires_siu'] else 'No'}")
print(f"  Adjuster Required: {'Yes' if risk['requires_adjuster'] else 'No'}")
print(f"  Imputed Fields: {feature_result['imputed_count']}/39")
print("\n✅ Pipeline test completed — 5 agents executed successfully.")


# ═══════════════════════════════════════════════════════════════
# 8. EXPORT FEATURES + PREDICTION TO CSV
# ═══════════════════════════════════════════════════════════════

prediction_dir = Path("prediction_folder")
prediction_dir.mkdir(exist_ok=True)

features_file = prediction_dir / "features.csv"

# Add prediction details
export_row = {
    "policy_number": parsed.get("policy_number"),
    "incident_date": parsed.get("incident_date"),
    "claim_amount": features.get("total_claim_amount"),
    "risk_score": risk["risk_score"],
    "risk_level": risk["risk_level"],
    "fraud_probability": fraud_result["fraud_probability"],
    "fraud_flag": fraud_result["fraud_flag"],
    **features
}

df = pd.DataFrame([export_row])

if features_file.exists():
    df.to_csv(features_file, mode='a', header=False, index=False)
else:
    df.to_csv(features_file, index=False)

print(f"\n📁 Features + prediction saved to {features_file.absolute()}")