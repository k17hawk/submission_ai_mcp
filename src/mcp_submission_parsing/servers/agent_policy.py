"""Agent 2: Policy Lookup with Multi-Sheet Joins - Optimized and Configurable"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PolicyAgent:
    """Looks up policy information from all sheets with proper joins"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize policy agent.
        
        Args:
            data_path: Path to Excel file. If None, reads from environment variable
                      POLICY_DATA_PATH or uses default.
        """
        if data_path is None:
            data_path = os.environ.get('POLICY_DATA_PATH', 
                                       '/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx')
        self.data_path = data_path
        self.policy_data = {}
        self._loaded = False
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all sheets once, build indexes, and clean up"""
        if self._loaded:
            return
        
        try:
            if not Path(self.data_path).exists():
                logger.error(f"Data file not found: {self.data_path}")
                self.policy_data = {}
                self._loaded = True
                return

            logger.info(f"Loading policy data from {self.data_path}")
            
            # Load all sheets – use low_memory=False to avoid mixed types
            sheets = {
                'policy': 'Policy_Data',
                'coverage': 'Coverage_Line',
                'customer': 'Customer_Master',
                'vehicle': 'Vehicle',
                'claim': 'Claim'
            }
            
            dfs = {}
            for key, sheet_name in sheets.items():
                try:
                    dfs[key] = pd.read_excel(self.data_path, sheet_name=sheet_name)
                    logger.debug(f"Loaded {sheet_name}: {len(dfs[key])} rows")
                except Exception as e:
                    logger.warning(f"Failed to load sheet '{sheet_name}': {e}")
                    dfs[key] = pd.DataFrame()  # empty fallback
            
            # Convert date columns to datetime once (vectorized)
            date_columns = {
                'policy': ['effective_date', 'expiration_date'],
                'customer': ['date_of_birth', 'customer_since_date'],
                'claim': ['claim_date', 'incident_date']
            }
            for df_name, cols in date_columns.items():
                if df_name in dfs and not dfs[df_name].empty:
                    for col in cols:
                        if col in dfs[df_name].columns:
                            dfs[df_name][col] = pd.to_datetime(dfs[df_name][col], errors='coerce')
            
            # Normalize string columns to avoid case mismatches
            if not dfs['policy'].empty and 'policy_status' in dfs['policy'].columns:
                dfs['policy']['policy_status'] = dfs['policy']['policy_status'].str.upper()
            
            # Build indexes for O(1) lookups
            coverage_idx = {}
            if not dfs['coverage'].empty:
                dfs['coverage']['policy_number'] = dfs['coverage']['policy_number'].astype(str)
                coverage_idx = dfs['coverage'].set_index('policy_number').to_dict('index')
            
            customer_idx = {}
            if not dfs['customer'].empty:
                dfs['customer']['customer_id'] = dfs['customer']['customer_id'].astype(str)
                customer_idx = dfs['customer'].set_index('customer_id').to_dict('index')
            
            vehicle_idx = {}
            if not dfs['vehicle'].empty:
                dfs['vehicle']['vehicle_id'] = dfs['vehicle']['vehicle_id'].astype(str)
                vehicle_idx = dfs['vehicle'].set_index('vehicle_id').to_dict('index')
            
            # Pre‑compute prior claims counts per customer
            prior_claims_counts = {}
            if not dfs['claim'].empty:
                dfs['claim']['customer_id'] = dfs['claim']['customer_id'].astype(str)
                prior_claims_counts = dfs['claim'].groupby('customer_id').size().to_dict()
            
            # Build policy data dictionary
            if dfs['policy'].empty:
                logger.error("Policy_Data sheet is empty or missing")
                self.policy_data = {}
                self._loaded = True
                return
            
            # Convert policy numbers to string once
            dfs['policy']['policy_number'] = dfs['policy']['policy_number'].astype(str)
            
            for _, policy_row in dfs['policy'].iterrows():
                policy_num = policy_row['policy_number']
                customer_id = str(policy_row.get('customer_id', ''))
                vehicle_id = str(policy_row.get('vehicle_id', ''))
                
                # Lookups
                cov = coverage_idx.get(policy_num, {})
                cust = customer_idx.get(customer_id, {})
                veh = vehicle_idx.get(vehicle_id, {})
                prior_claims = prior_claims_counts.get(customer_id, 0)
                
                # Determine active status
                policy_status = str(policy_row.get('policy_status', 'UNKNOWN'))
                is_active = policy_status == 'ACTIVE'
                
                # Prepare result dict
                result = {
                    'found': True,
                    'policy_number': policy_num,
                    
                    # Policy_Data
                    'policy_status': policy_status,
                    'effective_date': self._date_to_str(policy_row.get('effective_date')),
                    'expiration_date': self._date_to_str(policy_row.get('expiration_date')),
                    'customer_id': customer_id,
                    'insured_name': str(policy_row.get('insured_name', '')),
                    'insured_state': str(policy_row.get('insured_state', '')),
                    'total_annual_premium': float(policy_row.get('total_annual_premium', 0)),
                    'umbrella_limit': float(policy_row.get('umbrella_limit', 0)),
                    'vehicle_id': vehicle_id,
                    'is_active': is_active,
                    
                    # Coverage_Line (defaults)
                    'deductible': float(cov.get('deductible')) if cov.get('deductible') is not None else None,
                    'csl': str(cov.get('csl', '')) if cov.get('csl') else None,
                    'coverage_limit': float(cov.get('coverage_limit')) if cov.get('coverage_limit') is not None else None,
                    'coverage_code': str(cov.get('coverage_code', '')) if cov.get('coverage_code') else None,
                    'coverage_name': str(cov.get('coverage_name', '')) if cov.get('coverage_name') else None,
                    
                    # Customer_Master
                    'date_of_birth': self._date_to_str(cust.get('date_of_birth')),
                    'customer_since_date': self._date_to_str(cust.get('customer_since_date')),
                    'gender': str(cust.get('gender', '')) if cust.get('gender') else None,
                    'education_level': str(cust.get('education_level', '')) if cust.get('education_level') else None,
                    'occupation': str(cust.get('occupation', '')) if cust.get('occupation') else None,
                    'hobbies': str(cust.get('hobbies', '')) if cust.get('hobbies') else None,
                    'relationship_status': str(cust.get('relationship_status', '')) if cust.get('relationship_status') else None,
                    'credit_score': float(cust.get('credit_score')) if cust.get('credit_score') is not None else None,
                    'capital_gains': float(cust.get('capital_gains', 0)),
                    'capital_loss': float(cust.get('capital_loss', 0)),
                    'lifetime_claims': int(cust.get('lifetime_claims', 0)),
                    
                    # Vehicle
                    'make': str(veh.get('make', '')) if veh.get('make') else None,
                    'model': str(veh.get('model', '')) if veh.get('model') else None,
                    'year': int(veh.get('year')) if veh.get('year') is not None else None,
                    'telematics_enrolled': str(veh.get('telematics_enrolled', 'NO')).upper(),
                    'telematics_score': float(veh.get('telematics_score')) if veh.get('telematics_score') is not None else None,
                    'market_value': float(veh.get('market_value')) if veh.get('market_value') is not None else None,
                    
                    'prior_claims_count': prior_claims,
                    'errors': [],
                    'warnings': []
                }
                
                # Derived fields: months as customer, age, incident_in_policy_period (to be filled later)
                result['months_as_customer'] = self._compute_months_as_customer(result['effective_date'])
                result['age'] = self._compute_age(result['date_of_birth'])
                result['incident_in_policy_period'] = None  # Will be set when incident_date is provided
                
                self.policy_data[policy_num] = result
            
            # Free memory – delete DataFrames
            del dfs
            logger.info(f"✅ Loaded {len(self.policy_data)} policies with complete data joins")
            
        except Exception as e:
            logger.error(f"Error loading policy data: {e}", exc_info=True)
            self.policy_data = {}
        finally:
            self._loaded = True
    
    @staticmethod
    def _date_to_str(dt):
        """Convert datetime to string YYYY-MM-DD, return None if NaT or None"""
        if pd.isna(dt) or dt is None:
            return None
        if isinstance(dt, (datetime, pd.Timestamp)):
            return dt.strftime('%Y-%m-%d')
        return str(dt)
    
    @staticmethod
    def _compute_months_as_customer(effective_date_str):
        """Compute months between effective date and now"""
        if not effective_date_str:
            return None
        try:
            eff_date = pd.to_datetime(effective_date_str)
            if pd.isna(eff_date):
                return None
            months = (datetime.now() - eff_date).days // 30
            return max(0, months)
        except:
            return None
    
    @staticmethod
    def _compute_age(dob_str):
        """Compute age from date of birth"""
        if not dob_str:
            return None
        try:
            dob = pd.to_datetime(dob_str)
            if pd.isna(dob):
                return None
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return max(0, age)
        except:
            return None
    
    @staticmethod
    def _normalize_policy(val: str) -> str:
        """Ensure policy number has POL- prefix"""
        val = str(val).strip()
        if val.upper().startswith("POL-"):
            return val.upper()
        return f"POL-{val}"
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup policy with complete data"""
        
        policy_number = payload.get('policy_number')
        incident_date = payload.get('incident_date', '')
        incident_type = payload.get('incident_type', '')
        claim_amount = payload.get('claim_amount', 0)
        parsed_auto_make = payload.get('auto_make', '')  # from parser
        
        if not policy_number:
            return {
                'found': False,
                'policy_number': None,
                'errors': ['No policy number provided'],
                'warnings': []
            }
        
        policy_number = self._normalize_policy(str(policy_number))
        
        # Check if policy exists
        if policy_number not in self.policy_data:
            return {
                'found': False,
                'policy_number': policy_number,
                'errors': [f"Policy {policy_number} not found in database"],
                'warnings': []
            }
        
        # Get base policy data (copy to avoid modifying cached dict)
        result = self.policy_data[policy_number].copy()
        
        # Add vehicle make mismatch warning if auto_make provided
        if parsed_auto_make and result.get('make'):
            if parsed_auto_make.upper() != result['make'].upper():
                result['warnings'].append(
                    f"Claim vehicle make '{parsed_auto_make}' does not match policy vehicle '{result['make']}'"
                )
        
        # Validate incident date if provided
        if incident_date:
            try:
                inc_date = pd.to_datetime(incident_date)
                eff_date = pd.to_datetime(result.get('effective_date')) if result.get('effective_date') else None
                exp_date = pd.to_datetime(result.get('expiration_date')) if result.get('expiration_date') else None
                
                if pd.isna(inc_date):
                    result['incident_in_policy_period'] = None
                    result['warnings'].append("Incident date could not be parsed")
                elif pd.isna(eff_date) or pd.isna(exp_date):
                    result['incident_in_policy_period'] = None
                    result['warnings'].append("Policy effective/expiration dates missing")
                else:
                    if eff_date <= inc_date <= exp_date:
                        result['incident_in_policy_period'] = True
                        # Add warnings for near-boundary dates
                        days_to_expiry = (exp_date - inc_date).days
                        days_from_start = (inc_date - eff_date).days
                        if days_to_expiry <= 30:
                            result['warnings'].append(f"Incident occurred within {days_to_expiry} days of policy expiration")
                        if days_from_start <= 30:
                            result['warnings'].append(f"Incident occurred within {days_from_start} days of policy effective date")
                    else:
                        result['incident_in_policy_period'] = False
                        result['errors'].append(
                            f"Incident date {incident_date} outside policy period "
                            f"({result['effective_date']} to {result['expiration_date']})"
                        )
            except Exception as e:
                result['incident_in_policy_period'] = None
                result['warnings'].append(f"Could not validate incident date: {e}")
        
        # Check for missing critical data
        critical_fields = ['deductible', 'csl', 'credit_score', 'telematics_score']
        missing_critical = [f for f in critical_fields if result.get(f) is None]
        if missing_critical:
            result['warnings'].append(f"Missing critical data: {', '.join(missing_critical)}")
        
        # Add warning for low credit score or low telematics score
        if result.get('credit_score') is not None and result['credit_score'] < 500:
            result['warnings'].append(f"Low credit score: {result['credit_score']}")
        if result.get('telematics_score') is not None and result['telematics_score'] < 40:
            result['warnings'].append(f"Low telematics score: {result['telematics_score']}")
        
        return result