"""Agent 2: Policy Lookup and Verification"""

from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class PolicyAgent:
    """Looks up policy information and verifies coverage"""
    
    def __init__(self, data_path: str = None):
        self.policies = {}
        self.customers = {}
        self.data_path = data_path or "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
        self._load_data()
    
    def _load_data(self):
        """Load policy data from Excel"""
        try:
            if Path(self.data_path).exists():
                df = pd.read_excel(self.data_path, sheet_name='Policy_Data')
                for _, row in df.iterrows():
                    policy_number = str(row['policy_number'])
                    self.policies[policy_number] = {
                        'found': True,
                        'policy_number': policy_number,
                        'customer_id': str(row.get('customer_id', '')),
                        'policy_status': str(row.get('policy_status', 'Active')),
                        'effective_date': str(row.get('effective_date', '')),
                        'expiration_date': str(row.get('expiration_date', '')),
                        'insured_name': str(row.get('insured_name', '')),
                        'coverage_code': str(row.get('coverage_code', 'COLL')),
                        'coverage_limit': float(row.get('coverage_limit', 250000)),
                        'deductible': float(row.get('deductible', 500)),
                        'vehicle_make': str(row.get('vehicle_make', '')),
                        'vehicle_model': str(row.get('vehicle_model', '')),
                        'vehicle_year': int(row.get('vehicle_year', 2000)),
                        'vehicle_value': float(row.get('vehicle_value', 20000)),
                        'prior_claims': int(row.get('prior_claims', 0))
                    }
                print(f"✅ Loaded {len(self.policies)} policies")
            else:
                print(f"⚠️ Data file not found: {self.data_path}")
                self._create_sample_data()
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        sample_policy = {
            'found': True,
            'policy_number': 'POL-123456',
            'customer_id': 'CUST001',
            'policy_status': 'Active',
            'effective_date': '2024-01-01',
            'expiration_date': '2025-01-01',
            'insured_name': 'John Smith',
            'coverage_code': 'COLL',
            'coverage_limit': 250000.0,
            'deductible': 500.0,
            'vehicle_make': 'Toyota',
            'vehicle_model': 'Camry',
            'vehicle_year': 2020,
            'vehicle_value': 25000.0,
            'prior_claims': 1
        }
        self.policies['POL-123456'] = sample_policy
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup and verify policy"""
        
        policy_number = payload.get('policy_number')
        incident_date = payload.get('incident_date', '')
        incident_type = payload.get('incident_type', '')
        claim_amount = payload.get('claim_amount', 0)
        
        result = {
            'found': False,
            'policy_number': policy_number,
            'errors': [],
            'warnings': []
        }
        
        # Lookup policy
        policy = self.policies.get(policy_number)
        if not policy:
            result['errors'].append(f"Policy {policy_number} not found")
            return result
        
        result['found'] = True
        result.update(policy)
        
        # Check policy status
        if policy['policy_status'] != 'Active':
            result['errors'].append(f"Policy status is '{policy['policy_status']}'")
            result['is_active'] = False
        else:
            result['is_active'] = True
        
        # Check incident date within policy period
        try:
            inc_date = datetime.strptime(incident_date, '%Y-%m-%d')
            eff_date = datetime.strptime(policy['effective_date'], '%Y-%m-%d')
            exp_date = datetime.strptime(policy['expiration_date'], '%Y-%m-%d')
            
            if eff_date <= inc_date <= exp_date:
                result['incident_in_policy_period'] = True
            else:
                result['errors'].append("Incident date outside policy period")
                result['incident_in_policy_period'] = False
        except Exception as e:
            result['warnings'].append(f"Could not validate date: {e}")
        
        # Check coverage
        covered_incidents = ['Single Vehicle Collision', 'Multi-vehicle Collision']
        if incident_type in covered_incidents and policy['coverage_code'] not in ['COLL', 'COMP']:
            result['errors'].append(f"Coverage mismatch: {policy['coverage_code']}")
        
        # Check claim amount
        if claim_amount > policy['coverage_limit']:
            result['errors'].append(f"Claim amount ${claim_amount:,.2f} exceeds limit ${policy['coverage_limit']:,.2f}")
        
        return result