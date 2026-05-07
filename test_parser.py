# test_pipeline.py
"""
Serial integration test: Agent 1 → Agent 2 → Agent 3 → Agent 4
Picks a random active policy, generates a compatible claim,
runs the full pipeline (including feature building for ML),
and saves the generated features to CSV.
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

import pandas as pd

# ── Setup ────────────────────────────────────────────────────
parser = SubmissionParser()
DATA_PATH = "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
service = PolicyLookupService(DATA_PATH)

# Override coverage mapping so "Other" works with all coverage types
INCIDENT_TO_COVERAGE.update({
    "Other": ["COMP", "COLL", "LIAB", "UNINSMOT", "MED", "PIP"],
})


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


def generate_claim_amount(severity: str, vehicle_value: float) -> float:
    """Generate a plausible claim amount based on severity and vehicle value."""
    if severity == "Total Loss":
        return round(vehicle_value * random.uniform(0.85, 1.10), 2)
    elif severity == "Major Damage":
        return round(vehicle_value * random.uniform(0.25, 0.65), 2)
    elif severity == "Minor Damage":
        return round(vehicle_value * random.uniform(0.05, 0.25), 2)
    else:
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

# Pick compatible incident type
compatible_types = [
    inc for inc, covs in INCIDENT_TO_COVERAGE.items()
    if policy.coverage_code in covs
]
if not compatible_types:
    print(f"❌ Policy {policy.policy_number} has coverage '{policy.coverage_code}'")
    print(f"   No incident types map to this coverage.")
    sys.exit(1)

inc_type = random.choice(compatible_types)
severity = random.choice(["Minor Damage", "Major Damage"])
claim_amount = generate_claim_amount(severity, policy.vehicle_value)

# Build claim text (injects actual policy details)
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
print()

print("Parsed fields:")
for k, v in parsed.items():
    if v is not None:
        print(f"  {k}: {v}")
print(f"  complete: {result.is_complete()}")
print(f"  confidence: {result.extraction_confidence:.1%}")
print()

# Inject missing fields
if not parsed.get("total_claim_amount"):
    parsed["total_claim_amount"] = claim_amount
if not parsed.get("police_report_available"):
    parsed["police_report_available"] = "YES"
if not parsed.get("witnesses"):
    parsed["witnesses"] = 1
if not parsed.get("bodily_injuries"):
    parsed["bodily_injuries"] = 0

print(f"  ℹ️  Injected: total_claim_amount=${claim_amount:,.2f}, "
      f"police_report=YES, witnesses=1, bodily_injuries=0")
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

# 3b. Verify identity – use name to find the correct customer record
print("\n>>> verify_identity (by name)")
iresp = verify_identity(
    customer_name=policy.insured_name,
    policy_number=parsed["policy_number"]
)
print(f"Verified: {iresp['verified']} — {iresp['message']}")

# Use the customer_id from the identity check for accurate customer profile
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

# Fetch customer profile using the ID obtained from identity verification
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
        customer_profile = service.get_customer(customer_id) if customer_id else None
        print(f"\n⚠️  Customer {customer_id} not found in Customer_Master")

feature_result = build_feature_vector(parsed, policy_data, verification, customer_profile)

print(f"\nFeatures: {feature_result['feature_count']}/27")
print(f"Customer profile used: {feature_result['customer_profile_used']}")
print(f"Imputed count: {feature_result['imputed_count']}")
if feature_result['imputed_fields']:
    print(f"Imputed fields: {feature_result['imputed_fields']}")
print(f"Ready for ML: {feature_result['ready_for_ml']}")

features = feature_result['features']

# Verify the customer data flowed through
print(f"\nDemographics verification:")
print(f"  insured_sex: {features['insured_sex']} ← customer said: {customer_profile.get('insured_sex') if customer_profile else 'N/A'}")
print(f"  insured_education_level: {features['insured_education_level']} ← customer said: {customer_profile.get('insured_education_level') if customer_profile else 'N/A'}")
print(f"  insured_occupation: {features['insured_occupation']} ← customer said: {customer_profile.get('insured_occupation') if customer_profile else 'N/A'}")
print(f"  months_as_customer: {features['months_as_customer']} (from customer profile)")
print(f"  age: {features['age']} (from customer profile)")


# ═══════════════════════════════════════════════════════════════
# 6. PIPELINE SUMMARY
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
print(f"  ─────────────────────────────────────")
print(f"  Agent 1 — Parsing: {result.extraction_confidence:.1%} confidence, "
      f"{'complete' if result.is_complete() else 'incomplete'}")
print(f"  Agent 2 — Coverage: {'PASS' if cov.get('covered') else 'FAIL'}")
print(f"  Agent 3 — Risk: {risk['risk_score']} ({risk['risk_level']}) → "
      f"{risk['auto_decision'] or 'MANUAL REVIEW'}")
print(f"  Agent 4 — Features: {feature_result['feature_count']} columns ready for ML")
print(f"  ─────────────────────────────────────")
print(f"  SIU Required: {'Yes' if risk['requires_siu'] else 'No'}")
print(f"  Adjuster Required: {'Yes' if risk['requires_adjuster'] else 'No'}")
print(f"  Imputed Fields: {feature_result['imputed_count']}/27")
print("\n✅ Pipeline test completed — 4 agents executed successfully.")

# ═══════════════════════════════════════════════════════════════
# 7. EXPORT FEATURES TO CSV
# ═══════════════════════════════════════════════════════════════

prediction_dir = Path("prediction_folder")
prediction_dir.mkdir(exist_ok=True)

features_file = prediction_dir / "features.csv"

# Create a row with identifier columns + all 27 features
export_row = {
    "policy_number": parsed.get("policy_number"),
    "incident_date": parsed.get("incident_date"),
    "claim_amount": features.get("total_claim_amount"),
    "risk_score": risk["risk_score"],
    "risk_level": risk["risk_level"],
    **features
}

df = pd.DataFrame([export_row])

# Append or create CSV
if features_file.exists():
    df.to_csv(features_file, mode='a', header=False, index=False)
else:
    df.to_csv(features_file, index=False)

print(f"\n📁 Features saved to {features_file.absolute()}")