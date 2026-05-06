"""
MCP Server 2: Identity & Policy Lookup
Verifies customer, policy coverage, deductible, and temporal validity.
Loads real data from the generated Excel dataset.
"""

import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyInfo:
    policy_number: str
    customer_id: str
    policy_type: str
    policy_status: str
    effective_date: str
    expiration_date: str
    insured_name: str
    insured_dob: str
    insured_address: str
    insured_phone: str
    insured_email: str
    coverage_code: str
    coverage_name: str
    coverage_limit: float
    coverage_deductible: float
    coverage_premium: float
    vehicle_vin: str
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    vehicle_value: float
    prior_claims_count: int
    payment_status: str
    credit_score: int
    telematics_enrolled: str
    annual_mileage: int


@dataclass
class PolicyVerificationResult:
    found: bool
    policy_number: str
    is_active: bool = False
    policy_status: str = "Unknown"
    incident_in_policy_period: bool = False
    effective_date: str = ""
    expiration_date: str = ""
    customer_id: str = ""
    customer_name: str = ""
    coverage_code: str = ""
    coverage_name: str = ""
    coverage_limit: float = 0.0
    deductible: float = 0.0
    vehicle_make: str = ""
    vehicle_model: str = ""
    vehicle_year: int = 0
    vehicle_value: float = 0.0
    prior_claims: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY LOOKUP SERVICE (DATA-DRIVEN)
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyLookupService:
    """
    Loads policy data from the generated Excel file.
    Creates an in-memory index on policy_number for fast lookups.
    """
    
    def __init__(self, excel_path: str = None):
        # Initialize ALL instance variables in __init__
        self._policies: Dict[str, PolicyInfo] = {}
        self._customer_policies: Dict[str, List[str]] = {}
        self._customers: Dict[str, Dict[str, Any]] = {}
        
        if excel_path:
            self.load_from_excel(excel_path)
    
    def load_from_excel(self, excel_path: str) -> None:
        """Load policies AND customers from the generated dataset."""
        path = Path(excel_path)
        if not path.exists():
            print(f"Warning: {excel_path} not found. Using empty database.")
            return
        
        # ── Load Policy Data (primary) ────────────────────────────────────
        df_policy = pd.read_excel(excel_path, sheet_name='Policy_Data')
        
        for _, row in df_policy.iterrows():
            policy = PolicyInfo(
                policy_number=str(row['policy_number']),
                customer_id=str(row['customer_id']),
                policy_type=str(row['policy_type']),
                policy_status=str(row['policy_status']),
                effective_date=str(row['effective_date']),
                expiration_date=str(row['expiration_date']),
                insured_name=str(row['insured_name']),
                insured_dob=str(row['insured_dob']),
                insured_address=str(row['insured_address']),
                insured_phone=str(row['insured_phone']),
                insured_email=str(row['insured_email']),
                coverage_code=str(row['coverage_code']),
                coverage_name=str(row['coverage_name']),
                coverage_limit=float(row['coverage_limit']),
                coverage_deductible=float(row['coverage_deductible']),
                coverage_premium=float(row['coverage_premium']),
                vehicle_vin=str(row['vehicle_vin']),
                vehicle_make=str(row['vehicle_make']),
                vehicle_model=str(row['vehicle_model']),
                vehicle_year=int(row['vehicle_year']),
                vehicle_value=float(row['vehicle_value']),
                prior_claims_count=int(row['prior_claims_count']),
                payment_status=str(row['payment_status']),
                credit_score=int(row['credit_score']),
                telematics_enrolled=str(row['telematics_enrolled']),
                annual_mileage=int(row['annual_mileage']),
            )
            
            self._policies[policy.policy_number] = policy
            
            # Build customer → policies index
            cid = policy.customer_id
            if cid not in self._customer_policies:
                self._customer_policies[cid] = []
            if policy.policy_number not in self._customer_policies[cid]:
                self._customer_policies[cid].append(policy.policy_number)
        
        # ── Load Customer Master (for identity verification) ──────────────
        df_cust = pd.read_excel(excel_path, sheet_name='Customer_Master')
        
        for _, row in df_cust.iterrows():
            self._customers[str(row['customer_id'])] = {
                'customer_id': str(row['customer_id']),
                'customer_name': str(row['customer_name']),
                'customer_dob': str(row['customer_dob']),
                'customer_address': str(row['customer_address']),
                'customer_phone': str(row['customer_phone']),
                'customer_email': str(row['customer_email']),
                'credit_score': int(row['credit_score']),
                'total_policies': int(row['total_policies']),
                'active_policies': int(row['active_policies']),
                'first_policy_date': str(row['first_policy_date']),
                'lifetime_claims': int(row['lifetime_claims']),
                'customer_segment': str(row['customer_segment']),
            }
        
        print(f"Loaded {len(self._policies):,} policies and {len(self._customers):,} customers")
    
    def get_policy(self, policy_number: str) -> Optional[PolicyInfo]:
        """Look up a single policy."""
        return self._policies.get(str(policy_number))
    
    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Look up a customer by ID."""
        return self._customers.get(str(customer_id))
    
    def get_customer_policies(self, customer_id: str) -> List[PolicyInfo]:
        """Get all policies for a customer."""
        policy_numbers = self._customer_policies.get(str(customer_id), [])
        return [self._policies[pn] for pn in policy_numbers if pn in self._policies]
    
    def get_customer_total_claims(self, customer_id: str) -> int:
        """Get total prior claims across all policies for a customer."""
        policies = self.get_customer_policies(customer_id)
        return sum(p.prior_claims_count for p in policies)
    
    


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

INCIDENT_TO_COVERAGE = {
    "Single Vehicle Collision": ["COLL", "COMP"],
    "Multi-vehicle Collision": ["COLL", "LIAB"],
    "Parked Car": ["COMP"],
    "Vehicle Theft": ["COMP"],
    "Other": ["COMP", "COLL"],
}

COVERAGE_DESCRIPTIONS = {
    "LIAB": "Liability",
    "COMP": "Comprehensive",
    "COLL": "Collision",
    "UNINSMOT": "Uninsured Motorist",
    "MED": "Medical Payments",
    "PIP": "Personal Injury Protection",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
service = PolicyLookupService(DATA_PATH)


def lookup_policy(policy_number: str) -> Dict[str, Any]:
    """MCP Tool: Look up a policy by number."""
    policy = service.get_policy(policy_number)
    if policy:
        return {"found": True, "policy": asdict(policy)}
    return {"found": False, "error": f"Policy {policy_number} not found"}


def check_coverage(
    policy_number: str,
    incident_type: str,
    incident_date: str,
    claim_amount: float = 0.0,
) -> Dict[str, Any]:
    """
    MCP Tool: Check if the incident is covered by the policy.
    
    Returns:
        Dict with covered status, deductible, coverage type, and limits.
    """
    policy = service.get_policy(policy_number)
    
    if not policy:
        return {
            "covered": False,
            "reason": "Policy not found",
            "deductible": 0,
            "coverage_type": "",
            "limits": {},
        }
    
    # Check policy status
    if policy.policy_status != "Active":
        return {
            "covered": False,
            "reason": f"Policy is {policy.policy_status}",
            "deductible": policy.coverage_deductible,
            "coverage_type": policy.coverage_name,
            "limits": {"coverage_limit": policy.coverage_limit},
        }
    
    # Check date falls within policy period
    try:
        inc_date = datetime.strptime(incident_date, "%Y-%m-%d")
        eff_date = datetime.strptime(policy.effective_date, "%Y-%m-%d")
        exp_date = datetime.strptime(policy.expiration_date, "%Y-%m-%d")
        
        if not (eff_date <= inc_date <= exp_date):
            return {
                "covered": False,
                "reason": f"Incident date {incident_date} is outside policy period ({policy.effective_date} to {policy.expiration_date})",
                "deductible": policy.coverage_deductible,
                "coverage_type": policy.coverage_name,
                "limits": {"coverage_limit": policy.coverage_limit},
            }
    except (ValueError, TypeError):
        pass  # If we can't parse dates, proceed with coverage check anyway
    
    # Check coverage type matches incident type
    required_coverages = INCIDENT_TO_COVERAGE.get(incident_type, [])
    coverage_ok = not required_coverages or policy.coverage_code in required_coverages
    
    if not coverage_ok:
        return {
            "covered": False,
            "reason": (
                f"Policy covers '{policy.coverage_code}' ({policy.coverage_name}) "
                f"but incident type '{incident_type}' requires {required_coverages}"
            ),
            "deductible": policy.coverage_deductible,
            "coverage_type": policy.coverage_name,
            "limits": {"coverage_limit": policy.coverage_limit},
        }
    
    # Check claim amount vs limit
    if claim_amount > policy.coverage_limit:
        return {
            "covered": False,
            "reason": f"Claim amount ${claim_amount:,.2f} exceeds coverage limit ${policy.coverage_limit:,.2f}",
            "deductible": policy.coverage_deductible,
            "coverage_type": policy.coverage_name,
            "limits": {"coverage_limit": policy.coverage_limit},
        }
    
    # All checks passed
    return {
        "covered": True,
        "deductible": policy.coverage_deductible,
        "coverage_type": policy.coverage_name,
        "limits": {"coverage_limit": policy.coverage_limit},
        "effective_date": policy.effective_date,
        "expiration_date": policy.expiration_date,
        "policy_status": policy.policy_status,
    }

def verify_policy_for_claim(
    policy_number: str,
    incident_date: str,
    incident_type: str,
    claim_amount: float = 0.0,
    vehicle_make: str = "",
    vehicle_model: str = "",
) -> Dict[str, Any]:
    """
    MCP Tool: Full policy verification for a claim.
    Checks: policy exists, active, in period, coverage match, limit, vehicle.
    """
    result = PolicyVerificationResult(
        found=False,
        policy_number=policy_number,
    )
    
    policy = service.get_policy(policy_number)
    if not policy:
        result.errors.append(f"Policy {policy_number} not found in system")
        return asdict(result)
    
    result.found = True
    result.policy_status = policy.policy_status
    result.effective_date = policy.effective_date
    result.expiration_date = policy.expiration_date
    result.customer_id = policy.customer_id
    result.customer_name = policy.insured_name
    result.coverage_code = policy.coverage_code
    result.coverage_name = policy.coverage_name
    result.coverage_limit = policy.coverage_limit
    result.deductible = policy.coverage_deductible
    result.vehicle_make = policy.vehicle_make
    result.vehicle_model = policy.vehicle_model
    result.vehicle_year = policy.vehicle_year
    result.vehicle_value = policy.vehicle_value
    result.prior_claims = policy.prior_claims_count
    
    # Policy status
    if policy.policy_status != "Active":
        result.errors.append(f"Policy status is '{policy.policy_status}', not 'Active'")
    else:
        result.is_active = True
    
    # Date check
    try:
        inc_date = datetime.strptime(incident_date, "%Y-%m-%d")
        eff_date = datetime.strptime(policy.effective_date, "%Y-%m-%d")
        exp_date = datetime.strptime(policy.expiration_date, "%Y-%m-%d")
        
        if eff_date <= inc_date <= exp_date:
            result.incident_in_policy_period = True
        else:
            result.errors.append(
                f"Incident date {incident_date} is outside policy period "
                f"({policy.effective_date} to {policy.expiration_date})"
            )
    except (ValueError, TypeError):
        result.warnings.append("Could not parse dates for policy period check")
    
    # Coverage match
    required_coverages = INCIDENT_TO_COVERAGE.get(incident_type, [])
    if required_coverages and policy.coverage_code not in required_coverages:
        result.errors.append(
            f"Incident type '{incident_type}' requires {required_coverages} coverage, "
            f"but policy has '{policy.coverage_code}' ({policy.coverage_name})"
        )
    
    # Limit check
    if claim_amount > 0 and claim_amount > policy.coverage_limit:
        result.errors.append(
            f"Claim amount ${claim_amount:,.2f} exceeds coverage limit ${policy.coverage_limit:,.2f}"
        )
    
    # Vehicle match
    if vehicle_make and vehicle_make.lower() != policy.vehicle_make.lower():
        result.warnings.append(
            f"Claimed vehicle make '{vehicle_make}' differs from policy '{policy.vehicle_make}'"
        )
    
    # Prior claims
    if policy.prior_claims_count >= 3:
        result.warnings.append(
            f"Customer has {policy.prior_claims_count} prior claims on this policy"
        )
    
    total_customer_claims = service.get_customer_total_claims(policy.customer_id)
    if total_customer_claims >= 5:
        result.warnings.append(
            f"Customer has {total_customer_claims} total claims across all policies"
        )
    
    return asdict(result)


def verify_identity(
    customer_name: str = "",
    customer_id: str = "",
    policy_number: str = "",
) -> Dict[str, Any]:
    """MCP Tool: Verify customer identity against policy records."""
    result = {
        "verified": False,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "policies": [],
        "message": "",
    }
    
    if policy_number:
        policy = service.get_policy(policy_number)
        if not policy:
            result["message"] = "Policy not found"
            return result
        
        result["customer_id"] = policy.customer_id
        result["customer_name"] = policy.insured_name
        
        if customer_name:
            if customer_name.lower() in policy.insured_name.lower() or \
               policy.insured_name.lower() in customer_name.lower():
                result["verified"] = True
                result["message"] = "Name matches policyholder"
            else:
                result["message"] = f"Name does not match. Policyholder is {policy.insured_name}"
        else:
            result["verified"] = True
            result["message"] = "Policy found"
        
        result["policies"] = [policy_number]
    
    elif customer_id:
        # Use get_customer for richer info
        customer = service.get_customer(customer_id)
        policies = service.get_customer_policies(customer_id)
        if customer and policies:
            result["verified"] = True
            result["customer_name"] = customer["customer_name"]
            result["policies"] = [p.policy_number for p in policies]
            result["message"] = f"Found {len(policies)} policies for customer"
            result["total_prior_claims"] = sum(p.prior_claims_count for p in policies)
            result["lifetime_claims"] = customer["lifetime_claims"]
            result["customer_segment"] = customer["customer_segment"]
        else:
            result["message"] = "Customer not found"
    
    else:
        result["message"] = "Provide policy_number or customer_id"
    
    return result


def pre_fill_claim(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """MCP Tool: Merge parsed submission with looked-up policy data."""
    policy_number = parsed_data.get("policy_number")
    if not policy_number:
        return {"error": "No policy_number in parsed data"}
    
    policy = service.get_policy(policy_number)
    if not policy:
        return {"error": f"Policy {policy_number} not found"}
    
    verification = verify_policy_for_claim(
        policy_number=policy_number,
        incident_date=parsed_data.get("incident_date", ""),
        incident_type=parsed_data.get("incident_type", ""),
        claim_amount=parsed_data.get("total_claim_amount", 0) or 0,
        vehicle_make=parsed_data.get("auto_make", ""),
    )
    
    return {
        "policy": asdict(policy),
        "parsed_submission": parsed_data,
        "verification": verification,
        "status": "ready_for_risk_rules",
        "requires_manual_review": len(verification.get("errors", [])) > 0,
    }




# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("POLICY LOOKUP SERVICE — TEST")
    print("=" * 60)
    
    if service._policies:
        test_policy = list(service._policies.keys())[0]
        print(f"\nTesting with policy: {test_policy}")
        
        result = verify_policy_for_claim(
            policy_number=test_policy,
            incident_date="2023-05-12",
            incident_type="Single Vehicle Collision",
            claim_amount=4500.00,
        )
        
        print(f"Found: {result['found']}")
        print(f"Active: {result['is_active']}")
        print(f"Status: {result['policy_status']}")
        print(f"In policy period: {result['incident_in_policy_period']}")
        print(f"Coverage: {result['coverage_code']} ({result['coverage_name']})")
        print(f"Deductible: ${result['deductible']:,.2f}")
        print(f"Errors: {result['errors']}")
        print(f"Warnings: {result['warnings']}")
        
        # Test customer lookup
        cust_id = result['customer_id']
        print(f"\nCustomer ID: {cust_id}")
        cust_result = verify_identity(customer_id=cust_id)
        print(f"Customer verified: {cust_result['verified']}")
        print(f"Name: {cust_result['customer_name']}")
        print(f"Policies: {cust_result['policies']}")
        print(f"Segment: {cust_result.get('customer_segment', 'N/A')}")
    else:
        print("No data loaded. Make sure underwriting_50k_dataset.xlsx exists.")