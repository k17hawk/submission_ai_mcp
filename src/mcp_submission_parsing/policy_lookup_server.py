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
    insured_state: str
    coverage_code: str
    coverage_name: str
    coverage_limit: float
    coverage_deductible: float
    coverage_premium: float
    total_annual_premium: float
    policy_csl: str
    vehicle_vin: str
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    vehicle_value: float
    prior_claims_count: int        
    payment_status: str
    credit_score: int
    telematics_enrolled: str
    telematics_score: float
    annual_mileage: int
    umbrella_limit: float


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


class PolicyLookupService:
    def __init__(self, excel_path: str = None):
        self._policies: Dict[str, PolicyInfo] = {}
        self._customer_policies: Dict[str, List[str]] = {}
        self._customers: Dict[str, Dict[str, Any]] = {}
        self._vehicles: Dict[str, Dict[str, Any]] = {}
        self._coverage: Dict[str, Dict[str, Any]] = {}
        self._claims: List[Dict[str, Any]] = []   # for prior claims calculation

        if excel_path:
            self.load_from_excel(excel_path)

    def load_from_excel(self, excel_path: str) -> None:
        path = Path(excel_path)
        if not path.exists():
            print(f"Warning: {excel_path} not found.")
            return

        # Load Customer_Master
        df_cust = pd.read_excel(excel_path, sheet_name='Customer_Master')
        self._customers = {}
        for _, row in df_cust.iterrows():
            self._customers[str(row['customer_id'])] = {
                'customer_id': str(row['customer_id']),
                'customer_name': str(row.get('customer_name', '')),
                'customer_dob': str(row.get('date_of_birth', '')),
                'customer_address': str(row.get('customer_address', '')),
                'customer_phone': str(row.get('customer_phone', '')),
                'customer_email': str(row.get('customer_email', '')),
                'credit_score': int(row.get('credit_score', 650)),
                'total_policies': int(row.get('total_policies', 0)),
                'active_policies': int(row.get('active_policies', 0)),
                'first_policy_date': str(row.get('customer_since_date', '')),
                'lifetime_claims': int(row.get('lifetime_claims', 0)),
                'customer_segment': str(row.get('customer_segment', 'Standard')),
                'insured_sex': str(row.get('gender', 'MALE')),
                'insured_education_level': str(row.get('education_level', 'Bachelors')),
                'insured_occupation': str(row.get('occupation', 'other-service')),
                'insured_hobbies': str(row.get('hobbies', 'reading')),
                'insured_relationship': str(row.get('relationship_status', 'not-in-family')),
                'capital_gains': float(row.get('capital_gains', 0)),
                'capital_loss': float(row.get('capital_loss', 0)),
            }

        # Load Vehicle
        try:
            df_veh = pd.read_excel(excel_path, sheet_name='Vehicle')
            for _, row in df_veh.iterrows():
                vid = str(row['vehicle_id'])
                self._vehicles[vid] = {
                    'vehicle_id': vid,
                    'customer_id': str(row['customer_id']),
                    'vin': str(row.get('vin', '')),
                    'make': str(row.get('make', '')),
                    'model': str(row.get('model', '')),
                    'year': int(row.get('year', 0)),
                    'body_style': str(row.get('body_style', '')),
                    'color': str(row.get('color', '')),
                    'market_value': float(row.get('market_value', 0)),
                    'annual_mileage': int(row.get('annual_mileage', 0)),
                    'primary_use': str(row.get('primary_use', '')),
                    'telematics_enrolled': str(row.get('telematics_enrolled', 'No')),
                    'telematics_score': float(row.get('telematics_score', 70.0)),
                }
            print(f"  Vehicles loaded: {len(self._vehicles):,}")
        except ValueError:
            print("  Vehicle sheet not found — continuing without vehicle data")

        # Load Coverage_Line
        self._coverage = {}
        try:
            df_cov = pd.read_excel(excel_path, sheet_name='Coverage_Line')
            for _, row in df_cov.iterrows():
                pn = str(row['policy_number'])
                if pn not in self._coverage:
                    self._coverage[pn] = {
                        'coverage_code': str(row.get('coverage_code', '')),
                        'coverage_name': str(row.get('coverage_name', '')),
                        'deductible': int(row.get('deductible', 500)),
                        'coverage_limit': float(row.get('coverage_limit', 250000)),
                        'csl': str(row.get('csl', '100/300')),
                        'premium_for_line': float(row.get('premium_for_line', 0)),
                    }
            print(f"  Coverage lines loaded: {len(self._coverage):,}")
        except ValueError:
            print("  Coverage_Line sheet not found — using defaults")

        # Load Policy_Data
        df_policy = pd.read_excel(excel_path, sheet_name='Policy_Data')
        for _, row in df_policy.iterrows():
            pn = str(row['policy_number'])
            cov = self._coverage.get(pn, {})
            vid = str(row['vehicle_id'])
            veh = self._vehicles.get(vid, {})

            policy = PolicyInfo(
                policy_number=pn,
                customer_id=str(row['customer_id']),
                policy_type=str(row.get('policy_type', 'Auto')),
                policy_status=str(row.get('policy_status', 'Active')),
                effective_date=str(row['effective_date']),
                expiration_date=str(row['expiration_date']),
                insured_name=str(row.get('insured_name', '')),
                insured_dob='',
                insured_address=str(row.get('insured_address', '')),
                insured_phone=str(row.get('insured_phone', '')),
                insured_email=str(row.get('insured_email', '')),
                insured_state=str(row.get('insured_state', 'CA')),
                coverage_code=cov.get('coverage_code', 'LIAB'),
                coverage_name=cov.get('coverage_name', 'Liability'),
                coverage_limit=float(cov.get('coverage_limit', 250000)),
                coverage_deductible=int(cov.get('deductible', 500)),
                coverage_premium=float(cov.get('premium_for_line', 0)),
                total_annual_premium=float(row.get('total_annual_premium', 1200)),
                policy_csl=cov.get('csl', '100/300'),
                vehicle_vin=veh.get('vin', ''),
                vehicle_make=veh.get('make', ''),
                vehicle_model=veh.get('model', ''),
                vehicle_year=int(veh.get('year', 0)),
                vehicle_value=float(veh.get('market_value', 0)),
                prior_claims_count=0,   # will be set after loading Claims
                payment_status=str(row.get('payment_status', 'Current')),
                credit_score=0,
                telematics_enrolled=veh.get('telematics_enrolled', 'No'),
                telematics_score=float(veh.get('telematics_score', 70.0)),
                annual_mileage=int(veh.get('annual_mileage', 0)),
                umbrella_limit=float(row.get('umbrella_limit', 0)),
            )
            self._policies[pn] = policy

            cid = policy.customer_id
            if cid not in self._customer_policies:
                self._customer_policies[cid] = []
            if pn not in self._customer_policies[cid]:
                self._customer_policies[cid].append(pn)

        # Load Claims sheet for prior claims count
        try:
            df_claims = pd.read_excel(excel_path, sheet_name='Claim')
            self._claims = df_claims.to_dict('records')
            # Build prior claims count per policy (all claims in the dataset)
            claim_counts = df_claims.groupby('policy_number').size().to_dict()
            for pn, count in claim_counts.items():
                if pn in self._policies:
                    self._policies[pn].prior_claims_count = count
            print(f"  Claims loaded: {len(self._claims):,}")
        except ValueError:
            print("  Claim sheet not found — prior_claims_count remains 0")

        print(f"Loaded {len(self._policies):,} policies, {len(self._customers):,} customers, "
              f"{len(self._coverage):,} coverage records")

    def get_policy(self, policy_number: str) -> Optional[PolicyInfo]:
        return self._policies.get(str(policy_number))

    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        return self._customers.get(str(customer_id))

    def get_customer_policies(self, customer_id: str) -> List[PolicyInfo]:
        policy_numbers = self._customer_policies.get(str(customer_id), [])
        return [self._policies[pn] for pn in policy_numbers if pn in self._policies]

    def get_customer_total_claims(self, customer_id: str) -> int:
        policies = self.get_customer_policies(customer_id)
        return sum(p.prior_claims_count for p in policies)


# Mappings and MCP tools (unchanged from previous version except for DATA_PATH)
INCIDENT_TO_COVERAGE = {
    "Single Vehicle Collision": ["COLL", "COMP"],
    "Multi-vehicle Collision": ["COLL", "LIAB"],
    "Parked Car": ["COMP"],
    "Vehicle Theft": ["COMP"],
    "Other": ["COMP", "COLL"],
}

DATA_PATH = "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
service = PolicyLookupService(DATA_PATH)

def lookup_policy(policy_number: str) -> Dict[str, Any]:
    policy = service.get_policy(policy_number)
    if policy:
        return {"found": True, "policy": asdict(policy)}
    return {"found": False, "error": f"Policy {policy_number} not found"}

def check_coverage(policy_number: str, incident_type: str, incident_date: str, claim_amount: float = 0.0) -> Dict[str, Any]:
    policy = service.get_policy(policy_number)
    if not policy:
        return {"covered": False, "reason": "Policy not found", "deductible": 0}
    if policy.policy_status != "Active":
        return {"covered": False, "reason": f"Policy is {policy.policy_status}", "deductible": policy.coverage_deductible}
    try:
        inc_date = datetime.strptime(incident_date, "%Y-%m-%d")
        eff_date = datetime.strptime(policy.effective_date, "%Y-%m-%d")
        exp_date = datetime.strptime(policy.expiration_date, "%Y-%m-%d")
        if not (eff_date <= inc_date <= exp_date):
            return {"covered": False, "reason": "Incident date outside policy period", "deductible": policy.coverage_deductible}
    except:
        pass
    required_coverages = INCIDENT_TO_COVERAGE.get(incident_type, [])
    if required_coverages and policy.coverage_code not in required_coverages:
        return {"covered": False, "reason": f"Coverage mismatch: {policy.coverage_code}", "deductible": policy.coverage_deductible}
    if claim_amount > policy.coverage_limit:
        return {"covered": False, "reason": "Claim amount exceeds limit", "deductible": policy.coverage_deductible}
    return {"covered": True, "deductible": policy.coverage_deductible, "coverage_type": policy.coverage_name,
            "limits": {"coverage_limit": policy.coverage_limit}, "effective_date": policy.effective_date,
            "expiration_date": policy.expiration_date, "policy_status": policy.policy_status}

def verify_policy_for_claim(policy_number: str, incident_date: str, incident_type: str, claim_amount: float = 0.0, vehicle_make: str = "") -> Dict[str, Any]:
    result = PolicyVerificationResult(found=False, policy_number=policy_number)
    policy = service.get_policy(policy_number)
    if not policy:
        result.errors.append(f"Policy {policy_number} not found")
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
    if policy.policy_status != "Active":
        result.errors.append(f"Policy status is '{policy.policy_status}'")
    else:
        result.is_active = True
    try:
        inc_date = datetime.strptime(incident_date, "%Y-%m-%d")
        eff_date = datetime.strptime(policy.effective_date, "%Y-%m-%d")
        exp_date = datetime.strptime(policy.expiration_date, "%Y-%m-%d")
        if eff_date <= inc_date <= exp_date:
            result.incident_in_policy_period = True
        else:
            result.errors.append("Incident date outside policy period")
    except:
        pass
    required_coverages = INCIDENT_TO_COVERAGE.get(incident_type, [])
    if required_coverages and policy.coverage_code not in required_coverages:
        result.errors.append(f"Coverage mismatch")
    if claim_amount > policy.coverage_limit:
        result.errors.append("Claim amount exceeds limit")
    if vehicle_make and vehicle_make.lower() != policy.vehicle_make.lower():
        result.warnings.append("Vehicle make mismatch")
    return asdict(result)

def verify_identity(customer_name: str = "", customer_id: str = "", policy_number: str = "") -> Dict[str, Any]:
    result = {"verified": False, "customer_id": customer_id, "customer_name": customer_name, "policies": [], "message": ""}
    if policy_number:
        policy = service.get_policy(policy_number)
        if not policy:
            result["message"] = "Policy not found"
            return result
        result["customer_id"] = policy.customer_id
        result["customer_name"] = policy.insured_name
        if customer_name and (customer_name.lower() in policy.insured_name.lower() or policy.insured_name.lower() in customer_name.lower()):
            result["verified"] = True
            result["message"] = "Name matches policyholder"
        else:
            result["message"] = f"Name does not match. Policyholder is {policy.insured_name}"
        result["policies"] = [policy_number]
    elif customer_id:
        customer = service.get_customer(customer_id)
        policies = service.get_customer_policies(customer_id)
        if customer and policies:
            result["verified"] = True
            result["customer_name"] = customer["customer_name"]
            result["policies"] = [p.policy_number for p in policies]
            result["message"] = f"Found {len(policies)} policies"
            result["total_prior_claims"] = sum(p.prior_claims_count for p in policies)
            result["lifetime_claims"] = customer["lifetime_claims"]
            result["customer_segment"] = customer["customer_segment"]
        else:
            result["message"] = "Customer not found"
    else:
        result["message"] = "Provide policy_number or customer_id"
    return result

def pre_fill_claim(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
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