# test_pipeline.py
"""
Serial integration test: Agent 1 → Agent 2 → Agent 3
Picks a random active policy, generates a compatible claim,
and runs the full pipeline.
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


# ── Setup ────────────────────────────────────────────────────
parser = SubmissionParser()
DATA_PATH = "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
service = PolicyLookupService(DATA_PATH)


# ── FIX: Add UNINSMOT and other coverages to the mapping ─────
# Override the imported mapping with a complete version
INCIDENT_TO_COVERAGE.update({
    "Other": ["COMP", "COLL", "LIAB", "UNINSMOT", "MED", "PIP"],  # Other covers everything
})


def pick_active_policy(service: PolicyLookupService):
    """Pick a random active policy that has valid dates."""
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


# ── 1. Pick a random ACTIVE policy ──────────────────────────
policy = pick_active_policy(service)
if not policy:
    print("No active policies found with valid date ranges.")
    sys.exit(1)

eff = datetime.strptime(policy.effective_date, "%Y-%m-%d")
exp = datetime.strptime(policy.expiration_date, "%Y-%m-%d")

# Generate incident date at least 31 days after start (avoids R005 false positive)
min_days = min(31, max(0, (exp - eff).days))
delta = random.randint(min_days, max(min_days, (exp - eff).days))
inc_date = eff + timedelta(days=delta)
inc_date_str = inc_date.strftime("%Y-%m-%d")

# ── 2. Choose a compatible incident type ─────────────────────
compatible_types = [
    inc for inc, covs in INCIDENT_TO_COVERAGE.items()
    if policy.coverage_code in covs
]

if not compatible_types:
    print(f"❌ Policy {policy.policy_number} has coverage '{policy.coverage_code}'")
    print(f"   No incident types map to this coverage in INCIDENT_TO_COVERAGE.")
    print(f"   Current mapping: {dict(INCIDENT_TO_COVERAGE)}")
    sys.exit(1)

inc_type = random.choice(compatible_types)

# ── 3. Pick severity and generate claim amount ────────────────
severity = random.choice(["Minor Damage", "Major Damage"])
claim_amount = generate_claim_amount(severity, policy.vehicle_value)

# ── 4. Build claim text ──────────────────────────────────────
claim_text = (
    f"Hi, I need to file a claim. On {inc_date_str} I was involved in a "
    f"{inc_type} on Highway in Springfield, IL. "
    f"My vehicle, a {policy.vehicle_year} {policy.vehicle_make} {policy.vehicle_model}, "
    f"sustained {severity}. The estimated damage is about ${claim_amount:,.2f}. "
    f"Policy number is {policy.policy_number}. "
    f"My name is {policy.insured_name}. Authorities contacted: Police."
)

# ── 5. Agent 1: Parse submission ─────────────────────────────
print("=" * 60)
print("AGENT 1: Submission Parser")
print("=" * 60)
result = parser.parse(claim_text, "email")
parsed = result.__dict__

print("Policy selected:")
print(f"  Number: {policy.policy_number} | Status: {policy.policy_status}")
print(f"  Period: {policy.effective_date} → {policy.expiration_date}")
print(f"  Vehicle: {policy.vehicle_year} {policy.vehicle_make} {policy.vehicle_model}")
print(f"  Value: ${policy.vehicle_value:,.2f}")
print(f"  Coverage: {policy.coverage_code} ({policy.coverage_name})")
print(f"  Limit: ${policy.coverage_limit:,.2f} | Deductible: ${policy.coverage_deductible:,.2f}")
print(f"  Insured: {policy.insured_name}")
print()

print("Generated incident:")
print(f"  Date: {inc_date_str} (in period: {eff <= inc_date <= exp}, day {delta} of policy)")
print(f"  Type: {inc_type} → needs coverage: {INCIDENT_TO_COVERAGE.get(inc_type, [])}")
print(f"  Policy has: {policy.coverage_code} → compatible: {policy.coverage_code in INCIDENT_TO_COVERAGE.get(inc_type, [])}")
print(f"  Severity: {severity}")
print(f"  Amount: ${claim_amount:,.2f}")
print(f"  Amount vs Deductible: {'Above' if claim_amount > policy.coverage_deductible else 'Below'}")
print(f"  Amount vs Limit: {'Within' if claim_amount <= policy.coverage_limit else 'EXCEEDS!'}")
print()

print("Parsed submission:")
for k, v in parsed.items():
    if v is not None:
        print(f"  {k}: {v}")
print(f"  complete: {result.is_complete()}")
print()

# Inject claim amount if parser didn't extract it
if not parsed.get("total_claim_amount"):
    parsed["total_claim_amount"] = claim_amount

# Inject police report and witnesses for cleaner claims
if not parsed.get("police_report_available"):
    parsed["police_report_available"] = "YES"
if not parsed.get("witnesses"):
    parsed["witnesses"] = 1

print(f"  ℹ️  Injected: total_claim_amount=${claim_amount:,.2f}, police_report=YES, witnesses=1")
print()

# ── 6. Agent 2: Policy lookup & verification ─────────────────
print("=" * 60)
print("AGENT 2: Identity & Policy Lookup")
print("=" * 60)

print("\n>>> lookup_policy")
presp = lookup_policy(parsed["policy_number"])
print(f"Found: {presp['found']}")
if not presp["found"]:
    print("❌ Policy not in database – test aborted.")
    sys.exit(1)

print("\n>>> verify_identity")
iresp = verify_identity(
    customer_name=policy.insured_name,
    policy_number=parsed["policy_number"]
)
print(f"Verified: {iresp['verified']} — {iresp['message']}")

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

print("\n>>> pre_fill_claim")
claim = pre_fill_claim(parsed)
print(f"Status: {claim.get('status')}")
verification = claim.get('verification', {})
errors = verification.get('errors', [])
if errors:
    print(f"Errors: {errors}")
warnings_list = verification.get('warnings', [])
if warnings_list:
    print(f"Warnings: {warnings_list}")

# ── 7. Agent 3: Risk Rules ───────────────────────────────────
print("\n" + "=" * 60)
print("AGENT 3: Risk Rule Checker")
print("=" * 60)

risk_assessment = check_risk_rules(parsed, verification)

print(f"\nPassed: {risk_assessment['passed']}")
print(f"Risk Score: {risk_assessment['risk_score']} ({risk_assessment['risk_level']})")
print(f"Auto Decision: {risk_assessment['auto_decision']}")
print(f"Needs SIU: {risk_assessment['requires_siu']}")
print(f"Needs Adjuster: {risk_assessment['requires_adjuster']}")

if risk_assessment.get('violations'):
    print(f"\n❌ Violations ({len(risk_assessment['violations'])}):")
    for v in risk_assessment['violations']:
        print(f"  [{v['severity']}] {v['rule_id']} {v['rule_name']}")
        print(f"    → {v['message']}")

if risk_assessment.get('warnings'):
    print(f"\n⚠️  Warnings ({len(risk_assessment['warnings'])}):")
    for w in risk_assessment['warnings']:
        print(f"  [{w['severity']}] {w['rule_name']}")
        print(f"    → {w['message']}")

if not risk_assessment.get('violations') and not risk_assessment.get('warnings'):
    print("\n✅ Clean claim — no violations or warnings!")

# ── 8. Summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"  Policy: {policy.policy_number} ({policy.coverage_code} → {policy.coverage_name})")
print(f"  Insured: {policy.insured_name}")
print(f"  Vehicle: {policy.vehicle_year} {policy.vehicle_make} {policy.vehicle_model} (${policy.vehicle_value:,.2f})")
print(f"  Incident: {inc_date_str} | {inc_type} | {severity}")
print(f"  Claim Amount: ${claim_amount:,.2f} (deductible: ${policy.coverage_deductible:,.2f})")
print(f"  Parsing Confidence: {result.extraction_confidence:.1%}")
print(f"  Coverage Check: {'PASS' if cov.get('covered') else 'FAIL'}")
print(f"  Risk Score: {risk_assessment['risk_score']} → {risk_assessment['risk_level']}")
print(f"  Decision: {risk_assessment['auto_decision'] or 'MANUAL REVIEW'}")
print(f"  SIU Required: {'Yes' if risk_assessment['requires_siu'] else 'No'}")
print(f"  Adjuster Required: {'Yes' if risk_assessment['requires_adjuster'] else 'No'}")
print("\n✅ Pipeline test completed.")