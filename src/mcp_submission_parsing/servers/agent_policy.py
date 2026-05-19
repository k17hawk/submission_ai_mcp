"""Agent 2: Policy Lookup with Multi-Sheet Joins"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys


class PolicyAgent:
    """Looks up policy information from all sheets with proper joins"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
        self.policy_data = {}
        self.coverage_data = {}
        self.customer_data = {}
        self.vehicle_data = {}
        self._load_all_data()
    
    def _load_all_data(self):
        try:
            if not Path(self.data_path).exists():
                print(f"❌ Data file not found: {self.data_path}")
                return

            # Load all sheets once
            policy_df   = pd.read_excel(self.data_path, sheet_name='Policy_Data')
            coverage_df = pd.read_excel(self.data_path, sheet_name='Coverage_Line')
            customer_df = pd.read_excel(self.data_path, sheet_name='Customer_Master')
            vehicle_df  = pd.read_excel(self.data_path, sheet_name='Vehicle')
            claim_df    = pd.read_excel(self.data_path, sheet_name='Claim')

            # Index everything once — O(1) lookups instead of full scans
            coverage_df['policy_number'] = coverage_df['policy_number'].astype(str)
    
            coverage_idx = coverage_df.set_index('policy_number').groupby(level=0).first()

            customer_df['customer_id'] = customer_df['customer_id'].astype(str)
            customer_idx = customer_df.set_index('customer_id')

            vehicle_df['vehicle_id'] = vehicle_df['vehicle_id'].astype(str)
            vehicle_idx = vehicle_df.set_index('vehicle_id')

            # Count prior claims per customer once up front
            claim_df['customer_id'] = claim_df['customer_id'].astype(str)
            prior_claims_counts = claim_df.groupby('customer_id').size().to_dict()

            policy_df['policy_number'] = policy_df['policy_number'].astype(str)

            for _, policy_row in policy_df.iterrows():
                policy_num  = str(policy_row['policy_number'])
                customer_id = str(policy_row.get('customer_id', ''))
                vehicle_id  = str(policy_row.get('vehicle_id', ''))

                # O(1) lookups
                cov = coverage_idx.loc[policy_num]  if policy_num  in coverage_idx.index else None
                cust = customer_idx.loc[customer_id] if customer_id in customer_idx.index else None
                veh  = vehicle_idx.loc[vehicle_id]   if vehicle_id  in vehicle_idx.index  else None
                prior_claims = prior_claims_counts.get(customer_id, 0)

                self.policy_data[policy_num] = {
                    'found': True,
                    'policy_number': policy_num,

                    # Policy_Data
                    'policy_status':        str(policy_row.get('policy_status', 'Unknown')),
                    'effective_date':       str(policy_row.get('effective_date', '')),
                    'expiration_date':      str(policy_row.get('expiration_date', '')),
                    'customer_id':          customer_id,
                    'insured_name':         str(policy_row.get('insured_name', '')),
                    'insured_state':        str(policy_row.get('insured_state', '')),
                    'total_annual_premium': float(policy_row.get('total_annual_premium', 0)),
                    'umbrella_limit':       float(policy_row.get('umbrella_limit', 0)),
                    'vehicle_id':           vehicle_id,
                    'is_active':            str(policy_row.get('policy_status', '')).upper() == 'ACTIVE',

                    # Coverage_Line
                    'deductible':     float(cov['deductible'])     if cov is not None else None,
                    'csl':            str(cov['csl'])               if cov is not None else None,
                    'coverage_limit': float(cov['coverage_limit']) if cov is not None else None,
                    'coverage_code':  str(cov['coverage_code'])    if cov is not None else None,
                    'coverage_name':  str(cov['coverage_name'])    if cov is not None else None,

                    # Customer_Master
                    'date_of_birth':      str(cust['date_of_birth'])      if cust is not None else None,
                    'customer_since_date':str(cust['customer_since_date']) if cust is not None else None,
                    'gender':             str(cust['gender'])              if cust is not None else None,
                    'education_level':    str(cust['education_level'])     if cust is not None else None,
                    'occupation':         str(cust['occupation'])          if cust is not None else None,
                    'hobbies':            str(cust['hobbies'])             if cust is not None else None,
                    'relationship_status':str(cust['relationship_status']) if cust is not None else None,
                    'credit_score':       float(cust['credit_score'])      if cust is not None else None,
                    'capital_gains':      float(cust['capital_gains'])     if cust is not None else None,
                    'capital_loss':       float(cust['capital_loss'])      if cust is not None else None,
                    'lifetime_claims':    int(cust['lifetime_claims'])     if cust is not None else 0,

                    # Vehicle
                    'make':               str(veh['make'])               if veh is not None else None,
                    'model':              str(veh['model'])               if veh is not None else None,
                    'year':               int(veh['year'])               if veh is not None else None,
                    'telematics_enrolled':str(veh['telematics_enrolled']) if veh is not None else 'NO',
                    'telematics_score':   float(veh['telematics_score']) if veh is not None else None,
                    'market_value':       float(veh['market_value'])     if veh is not None else None,

                    'prior_claims_count': prior_claims,
                    'errors':   [],
                    'warnings': []
                }

                # Derived fields
                eff = self.policy_data[policy_num]['effective_date']
                if eff and eff != 'nan':
                    try:
                        eff_date = datetime.strptime(eff, '%Y-%m-%d')
                        self.policy_data[policy_num]['months_as_customer'] = (datetime.now() - eff_date).days // 30
                    except:
                        self.policy_data[policy_num]['months_as_customer'] = None

                dob = self.policy_data[policy_num]['date_of_birth']
                if dob and dob != 'nan':
                    try:
                        dob_date = datetime.strptime(dob, '%Y-%m-%d')
                        self.policy_data[policy_num]['age'] = (datetime.now() - dob_date).days // 365
                    except:
                        self.policy_data[policy_num]['age'] = None

            print(f"✅ Loaded {len(self.policy_data)} policies with complete data joins")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()

    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup policy with complete data"""
        
        policy_number = payload.get('policy_number')
        incident_date = payload.get('incident_date', '')
        incident_type = payload.get('incident_type', '')
        claim_amount = payload.get('claim_amount', 0)
        
        # Check if policy exists
        if not policy_number or policy_number not in self.policy_data:
            return {
                'found': False,
                'policy_number': policy_number or 'UNKNOWN',
                'errors': [f"Policy {policy_number} not found in database"],
                'warnings': []
            }
        
        # Get complete policy data
        result = self.policy_data[policy_number].copy()
        
        # Validate incident date if provided
        if incident_date:
            try:
                inc_date = datetime.strptime(incident_date, '%Y-%m-%d')
                eff_date = datetime.strptime(result['effective_date'], '%Y-%m-%d') if result['effective_date'] and result['effective_date'] != 'nan' else None
                exp_date = datetime.strptime(result['expiration_date'], '%Y-%m-%d') if result['expiration_date'] and result['expiration_date'] != 'nan' else None
                
                if eff_date and exp_date:
                    if eff_date <= inc_date <= exp_date:
                        result['incident_in_policy_period'] = True
                    else:
                        result['incident_in_policy_period'] = False
                        result['errors'].append(f"Incident date {incident_date} outside policy period ({result['effective_date']} to {result['expiration_date']})")
                else:
                    result['incident_in_policy_period'] = None
                    result['warnings'].append("Policy dates not available")
            except Exception as e:
                result['incident_in_policy_period'] = None
                result['warnings'].append(f"Could not validate incident date: {e}")
        
        # Check for missing critical data
        critical_fields = ['deductible', 'csl', 'credit_score', 'telematics_score']
        missing_critical = [f for f in critical_fields if result.get(f) is None]
        if missing_critical:
            result['warnings'].append(f"Missing critical data: {', '.join(missing_critical)}")
        
        return result