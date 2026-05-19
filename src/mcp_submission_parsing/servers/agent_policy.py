"""Agent 2: Policy Lookup with Multi-Sheet Joins"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys
from src.mcp_submission_parsing.config.logger_config import get_logger  
logger = get_logger('policy agent')

class PolicyAgent:
    """Looks up policy information from all sheets with proper joins"""
    
    def __init__(self, data_path: str = None):
        logger.info("Initializing PolicyAgent")
        self.data_path = data_path or "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"
        logger.debug(f"Data path set to: {self.data_path}")
        self.policy_data = {}
        self.coverage_data = {}
        self.customer_data = {}
        self.vehicle_data = {}
        logger.info("Starting data load from Excel file")
        self._load_all_data()
        logger.info("PolicyAgent initialization complete")
    
    def _load_all_data(self):
        logger.info("Beginning data loading from all sheets")
        try:
            if not Path(self.data_path).exists():
                logger.error(f"❌ Data file not found: {self.data_path}")
                return

            logger.info(f"Loading Excel file: {self.data_path}")
            # Load all sheets once
            policy_df   = pd.read_excel(self.data_path, sheet_name='Policy_Data')
            logger.info(f"Loaded Policy_Data sheet: {len(policy_df)} rows")
            
            coverage_df = pd.read_excel(self.data_path, sheet_name='Coverage_Line')
            logger.info(f"Loaded Coverage_Line sheet: {len(coverage_df)} rows")
            
            customer_df = pd.read_excel(self.data_path, sheet_name='Customer_Master')
            logger.info(f"Loaded Customer_Master sheet: {len(customer_df)} rows")
            
            vehicle_df  = pd.read_excel(self.data_path, sheet_name='Vehicle')
            logger.info(f"Loaded Vehicle sheet: {len(vehicle_df)} rows")
            
            claim_df    = pd.read_excel(self.data_path, sheet_name='Claim')
            logger.info(f"Loaded Claim sheet: {len(claim_df)} rows")

            # Index everything once — O(1) lookups instead of full scans
            logger.debug("Creating indexes for fast lookups")
            coverage_df['policy_number'] = coverage_df['policy_number'].astype(str)
            coverage_idx = coverage_df.set_index('policy_number').groupby(level=0).first()
            logger.debug(f"Coverage index created with {len(coverage_idx)} unique policies")

            customer_df['customer_id'] = customer_df['customer_id'].astype(str)
            customer_idx = customer_df.set_index('customer_id').groupby(level=0).first()
            logger.debug(f"Customer index created with {len(customer_idx)} unique customers")

            vehicle_df['vehicle_id'] = vehicle_df['vehicle_id'].astype(str)
            vehicle_idx  = vehicle_df.set_index('vehicle_id').groupby(level=0).first()
            logger.debug(f"Vehicle index created with {len(vehicle_idx)} unique vehicles")

            # Count prior claims per customer once up front
            logger.debug("Calculating prior claims counts")
            claim_df['customer_id'] = claim_df['customer_id'].astype(str)
            prior_claims_counts = claim_df.groupby('customer_id').size().to_dict()
            logger.debug(f"Prior claims calculated for {len(prior_claims_counts)} customers")

            policy_df['policy_number'] = policy_df['policy_number'].astype(str)
            logger.info(f"Processing {len(policy_df)} policies")

            processed_count = 0
            error_count = 0
            
            for _, policy_row in policy_df.iterrows():
                try:
                    policy_num  = str(policy_row['policy_number'])
                    customer_id = str(policy_row.get('customer_id', ''))
                    vehicle_id  = str(policy_row.get('vehicle_id', ''))

                    # O(1) lookups
                    cov = coverage_idx.loc[policy_num]  if policy_num  in coverage_idx.index else None
                    if cov is None:
                        logger.debug(f"Policy {policy_num}: No coverage data found")
                    
                    cust = customer_idx.loc[customer_id] if customer_id in customer_idx.index else None
                    if cust is None and customer_id:
                        logger.debug(f"Policy {policy_num}: No customer data for ID {customer_id}")
                    
                    veh  = vehicle_idx.loc[vehicle_id]   if vehicle_id  in vehicle_idx.index  else None
                    if veh is None and vehicle_id:
                        logger.debug(f"Policy {policy_num}: No vehicle data for ID {vehicle_id}")
                    
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
                        except Exception as e:
                            logger.warning(f"Policy {policy_num}: Error parsing effective date '{eff}': {e}")
                            self.policy_data[policy_num]['months_as_customer'] = None

                    dob = self.policy_data[policy_num]['date_of_birth']
                    if dob and dob != 'nan':
                        try:
                            dob_date = datetime.strptime(dob, '%Y-%m-%d')
                            self.policy_data[policy_num]['age'] = (datetime.now() - dob_date).days // 365
                        except Exception as e:
                            logger.warning(f"Policy {policy_num}: Error parsing date of birth '{dob}': {e}")
                            self.policy_data[policy_num]['age'] = None
                    
                    processed_count += 1
                    if processed_count % 5000 == 0:
                        logger.info(f"Processed {processed_count}/{len(policy_df)} policies")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing policy row: {e}", exc_info=True)
                    continue

            logger.info(f"✅ Loaded {processed_count} policies with complete data joins ({error_count} errors)")

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            
    @staticmethod
    def _normalize_policy(val: str) -> str:
        logger.debug(f"Normalizing policy number: {val}")
        val = str(val).strip()
        result = val if val.startswith("POL-") else f"POL-{val}"
        logger.debug(f"Normalized policy number: {result}")
        return result

    @staticmethod  
    def _normalize_customer(val: str) -> str:
        logger.debug(f"Normalizing customer ID: {val}")
        val = str(val).strip()
        result = val if val.startswith("CUST-") else f"CUST-{val}"
        logger.debug(f"Normalized customer ID: {result}")
        return result
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup policy with complete data"""
        logger.info(f"Processing policy lookup request with payload: {payload}")
        
        policy_number = payload.get('policy_number')
        incident_date = payload.get('incident_date', '')
        incident_type = payload.get('incident_type', '')
        claim_amount  = payload.get('claim_amount', 0)

        logger.debug(f"Extracted - Policy: {policy_number}, Incident Date: {incident_date}, Type: {incident_type}, Amount: {claim_amount}")

        if policy_number:
            policy_number = self._normalize_policy(str(policy_number))
            logger.info(f"Normalized policy number: {policy_number}")

        # Check if policy exists
        if not policy_number or policy_number not in self.policy_data:
            logger.warning(f"Policy {policy_number} not found in database")
            return {
                'found': False,
                'policy_number': policy_number or 'UNKNOWN',
                'errors': [f"Policy {policy_number} not found in database"],
                'warnings': []
            }

        logger.info(f"Policy {policy_number} found, retrieving data")
        # Get complete policy data
        result = self.policy_data[policy_number].copy()
        logger.debug(f"Retrieved policy data for {policy_number}: Status={result.get('policy_status')}, Active={result.get('is_active')}")

        # Validate incident date if provided
        if incident_date:
            logger.info(f"Validating incident date {incident_date} for policy {policy_number}")
            try:
                inc_date = datetime.strptime(incident_date, '%Y-%m-%d')
                eff_date = datetime.strptime(result['effective_date'], '%Y-%m-%d') if result['effective_date'] and result['effective_date'] != 'nan' else None
                exp_date = datetime.strptime(result['expiration_date'], '%Y-%m-%d') if result['expiration_date'] and result['expiration_date'] != 'nan' else None

                if eff_date and exp_date:
                    if eff_date <= inc_date <= exp_date:
                        result['incident_in_policy_period'] = True
                        logger.info(f"Incident date {incident_date} is within policy period")
                    else:
                        result['incident_in_policy_period'] = False
                        error_msg = f"Incident date {incident_date} outside policy period ({result['effective_date']} to {result['expiration_date']})"
                        result['errors'].append(error_msg)
                        logger.warning(f"Policy {policy_number}: {error_msg}")
                else:
                    result['incident_in_policy_period'] = None
                    warning_msg = "Policy dates not available"
                    result['warnings'].append(warning_msg)
                    logger.warning(f"Policy {policy_number}: {warning_msg}")
            except Exception as e:
                result['incident_in_policy_period'] = None
                warning_msg = f"Could not validate incident date: {e}"
                result['warnings'].append(warning_msg)
                logger.error(f"Policy {policy_number}: {warning_msg}", exc_info=True)

        # Check for missing critical data
        critical_fields = ['deductible', 'csl', 'credit_score', 'telematics_score']
        missing_critical = [f for f in critical_fields if result.get(f) is None]
        if missing_critical:
            warning_msg = f"Missing critical data: {', '.join(missing_critical)}"
            result['warnings'].append(warning_msg)
            logger.warning(f"Policy {policy_number}: {warning_msg}")

        logger.info(f"Policy lookup complete for {policy_number}: Found={result['found']}, Errors={len(result['errors'])}, Warnings={len(result['warnings'])}")
        return result