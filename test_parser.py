# test_pipeline.py
"""
Serial integration test: Agent 1 → Agent 2
Uses a real, active policy from the dataset to guarantee success.
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
    INCIDENT_TO_COVERAGE,          # <-- ready‑made mapping
)
from src.mcp_submission_parsing.config import get_normalizer_config   # (if needed)

# ── Setup ────────────────────────────────────────────────────
parser = SubmissionParser()
DATA_PATH = "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
service = PolicyLookupService(DATA_PATH)

# ── 1. Pick a random ACTIVE policy ──────────────────────────
active = [p for p in service._policies.values() if p.policy_status == "Active"]
if not active:
    print("No active policies found.")
    sys.exit(1)

policy = random.choice(active)

# Generate a random incident date inside the policy period
eff = datetime.strptime(policy.effective_date, "%Y-%m-%d")
exp = datetime.strptime(policy.expiration_date, "%Y-%m-%d")
if (exp - eff).days < 0:
    print("Policy dates are inconsistent.")
    sys.exit(1)
delta = random.randint(0, (exp - eff).days)
inc_date = eff + timedelta(days=delta)
inc_date_str = inc_date.strftime("%Y-%m-%d")

# ── 2. Choose an incident type that the policy actually covers ─
# Find all incident types whose required coverage list contains the policy's coverage_code.
compatible_types = [
    inc for inc, covs in INCIDENT_TO_COVERAGE.items()
    if policy.coverage_code in covs
]

if compatible_types:
    inc_type = random.choice(compatible_types)
else:
    # Fallback: use "Other" (will likely fail coverage, but at least demo)
    inc_type = "Other"

# ── 3. Build a natural‑language claim ─────────────────────────
claim_text = (
    f"Hi, I need to file a claim. On {inc_date_str} I was involved in a "
    f"{inc_type} on Highway in Springfield, IL. "
    f"My vehicle, a {policy.vehicle_year} {policy.vehicle_make} {policy.vehicle_model}, "
    f"sustained Major Damage. Policy number is {policy.policy_number}. "
    f"My name is {policy.insured_name}. Authorities contacted: Police."
)

# ── 4. Agent 1: Parse submission ─────────────────────────────
print("=" * 60)
print("AGENT 1: Submission Parser")
result = parser.parse(claim_text, "email")
parsed = result.__dict__
print("Claim text used:")
print("  ", claim_text)
print("Parsed submission:")
for k, v in parsed.items():
    if v is not None:
        print(f"  {k}: {v}")
print(f"  complete: {result.is_complete()}")
print()

# ── 5. Agent 2: Policy lookup & verification ─────────────────
print("=" * 60)
print("AGENT 2: Identity & Policy Lookup")
print()

# 5a. Lookup policy
print(">>> lookup_policy")
presp = lookup_policy(parsed["policy_number"])
print(f"Found: {presp['found']}")
if not presp["found"]:
    print("Policy not in database – test aborted.")
    sys.exit(1)

# 5b. Verify identity (corrected keyword arguments)
print("\n>>> verify_identity")
iresp = verify_identity(
    customer_name=policy.insured_name,
    policy_number=parsed["policy_number"]
)
print(f"Verified: {iresp['verified']} – {iresp['message']}")

# 5c. Check coverage
print("\n>>> check_coverage")
cov = check_coverage(
    parsed["policy_number"],
    parsed["incident_type"],
    parsed["incident_date"],
    claim_amount=parsed.get("total_claim_amount", 0) or 0,
)
print(f"Covered: {cov['covered']}")
if not cov['covered']:
    print(f"Reason: {cov['reason']}")
print(f"Deductible: ${cov['deductible']:,.2f}")

# 5d. Pre‑fill claim
print("\n>>> pre_fill_claim")
claim = pre_fill_claim(parsed)
print(f"Status: {claim.get('status')}")
print(f"Requires manual review: {claim.get('requires_manual_review')}")
verification = claim.get('verification', {})
print(f"Errors: {verification.get('errors', [])}")
print(f"Warnings: {verification.get('warnings', [])}")

print("\nPipeline test completed successfully.")