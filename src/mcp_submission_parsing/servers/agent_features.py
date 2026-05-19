"""Agent 4: Feature Builder for ML - aligned with model expectations"""

from typing import Dict, Any, List
from datetime import datetime
from src.mcp_submission_parsing.common.transformer import QuestionMarkToNaN
from src.mcp_submission_parsing.config.logger_config import get_logger

logger = get_logger('feature agent')

class FeatureAgent:
    """Builds feature vector for fraud detection model"""
    
    def __init__(self):
        logger.info("Initializing FeatureAgent")
        
        # Full list of features expected by the trained model
        self.feature_columns = [
            'months_as_customer', 'age', 'insured_sex', 'insured_education_level',
            'insured_occupation', 'policy_state', 'policy_deductable', 
            'policy_annual_premium', 'credit_score', 'telematics_score',
            'incident_type', 'collision_type', 'incident_severity',
            'authorities_contacted', 'incident_hour_of_the_day',
            'number_of_vehicles_involved', 'property_damage', 'bodily_injuries',
            'witnesses', 'police_report_available', 'total_claim_amount',
            'prior_claims_count', 'auto_make', 'auto_year', 'injury_claim',
            'property_claim', 'vehicle_claim',
            # Additional columns required by model:
            'policy_csl', 'incident_state', 'umbrella_limit', 'capital_gains',
            'incident_near_boundary', 'capital_loss', 'insured_hobbies',
            'incident_location', 'insured_relationship'
        ]
        
        logger.debug(f"Feature agent initialized with {len(self.feature_columns)} expected features")
        
        self.defaults = {
            'credit_score': 650,
            'telematics_score': 70.0,
            'policy_deductable': 500,
            'policy_annual_premium': 1200.0,
            'prior_claims_count': 0,
            'bodily_injuries': 0,
            'witnesses': 0,
            'injury_claim': 0.0,
            'property_claim': 0.0,
            'vehicle_claim': 0.0,
            'policy_csl': '100/300',
            'umbrella_limit': 0.0,
            'capital_gains': 0.0,
            'capital_loss': 0.0,
            'incident_near_boundary': 0,
            'insured_hobbies': 'Unknown',
            'insured_relationship': 'Unknown',
            'incident_location': 'Other',
            'incident_state': 'Unknown',
            'insured_sex': 'Unknown',
            'insured_education_level': 'Unknown',
            'insured_occupation': 'Unknown',
            'policy_state': 'Unknown',
        }
        
        logger.debug(f"Default values configured for {len(self.defaults)} fields")
        
        # Mapping for incident_location to valid categories
        self.location_mapping = {
            'i-95': 'Interstate',
            'i-': 'Interstate',
            'highway': 'Highway',
            'freeway': 'Highway',
            'expressway': 'Highway',
            'interstate': 'Interstate',
            'local road': 'Local Road',
            'street': 'Local Road',
            'parking lot': 'Parking Lot',
            'garage': 'Parking Lot',
            'residential street': 'Residential Street',
            'residential': 'Residential Street',
            'neighborhood': 'Residential Street',
        }
        
        logger.info("FeatureAgent initialization complete")
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build feature vector"""
        logger.info("Starting feature vector construction")
        
        parsed = payload.get('parsed', {})
        policy = payload.get('policy', {})
        verification = payload.get('verification', {})
        
        logger.debug(f"Input data sizes - parsed: {len(parsed)} fields, policy: {len(policy)} fields, verification: {len(verification)} fields")
        
        features = {}
        imputed_fields = []
        
        # Build features from available data
        logger.info(f"Processing {len(self.feature_columns)} feature columns")
        
        for idx, col in enumerate(self.feature_columns, 1):
            logger.debug(f"Processing feature {idx}/{len(self.feature_columns)}: {col}")
            
            value = None
            source = None
            
            # Map each column to its source
            if col == 'policy_csl':
                value = verification.get('csl')
                source = 'verification.csl'
                logger.debug(f"Mapping {col} to verification.csl: {value}")
                
            elif col == 'incident_state':
                # Try parsed, then verification.insured_state
                value = parsed.get('incident_state') or verification.get('insured_state')
                source = 'parsed.incident_state or verification.insured_state'
                logger.debug(f"Mapping {col} to incident_state/insured_state: {value}")
                
            elif col == 'umbrella_limit':
                value = verification.get('umbrella_limit')
                source = 'verification.umbrella_limit'
                logger.debug(f"Mapping {col} to verification.umbrella_limit: {value}")
                
            elif col == 'capital_gains':
                value = verification.get('capital_gains')
                source = 'verification.capital_gains'
                logger.debug(f"Mapping {col} to verification.capital_gains: {value}")
                
            elif col == 'capital_loss':
                value = verification.get('capital_loss')
                source = 'verification.capital_loss'
                logger.debug(f"Mapping {col} to verification.capital_loss: {value}")
                
            elif col == 'insured_hobbies':
                value = verification.get('hobbies')
                source = 'verification.hobbies'
                logger.debug(f"Mapping {col} to verification.hobbies: {value}")
                
            elif col == 'insured_relationship':
                value = verification.get('relationship_status')
                source = 'verification.relationship_status'
                logger.debug(f"Mapping {col} to verification.relationship_status: {value}")
                
            elif col == 'incident_location':
                raw_location = parsed.get('incident_location', '')
                logger.debug(f"Raw incident location: '{raw_location}'")
                # Normalize to one of the valid categories
                if raw_location:
                    norm = raw_location.lower().strip()
                    for key, mapped in self.location_mapping.items():
                        if key in norm:
                            value = mapped
                            logger.debug(f"Mapped '{raw_location}' to '{value}' via key '{key}'")
                            break
                    if not value:
                        value = raw_location.title()  # fallback
                        logger.debug(f"No mapping found for '{raw_location}', using title case: '{value}'")
                source = 'parsed.incident_location'
                
            elif col == 'incident_near_boundary':
                # Compute from dates
                value = self._compute_near_boundary(parsed, verification)
                source = 'derived'
                logger.debug(f"Computed incident_near_boundary: {value}")
                
            else:
                # For other columns, try parsed, then verification, then policy
                if col in parsed and parsed[col] is not None:
                    value = parsed[col]
                    source = 'parsed'
                    logger.debug(f"Found {col} in parsed data: {value}")
                elif col in verification and verification[col] is not None:
                    value = verification[col]
                    source = 'verification'
                    logger.debug(f"Found {col} in verification data: {value}")
                elif col in policy and policy[col] is not None:
                    value = policy[col]
                    source = 'policy'
                    logger.debug(f"Found {col} in policy data: {value}")
                else:
                    logger.debug(f"{col} not found in parsed, verification, or policy")
            
            # If still None, use default
            if value is None:
                value = self.defaults.get(col)
                if value is not None:
                    imputed_fields.append(col)
                    logger.debug(f"Using default value for {col}: {value} (default from config)")
                else:
                    logger.warning(f"No default value found for {col}, setting to None")
            
            # Type conversion
            if col in ['auto_year', 'bodily_injuries', 'witnesses', 
                       'prior_claims_count', 'incident_hour_of_the_day', 'incident_near_boundary']:
                try:
                    original_value = value
                    value = int(value) if value else 0
                    if original_value != value:
                        logger.debug(f"Converted {col} from {original_value} to int: {value}")
                except Exception as e:
                    logger.warning(f"Error converting {col} to int (value: {value}): {e}, setting to 0")
                    value = 0
                    
            elif col in ['total_claim_amount', 'injury_claim', 'property_claim', 
                         'vehicle_claim', 'policy_annual_premium', 'policy_deductable',
                         'umbrella_limit', 'capital_gains', 'capital_loss']:
                try:
                    original_value = value
                    value = float(value) if value else 0.0
                    if original_value != value:
                        logger.debug(f"Converted {col} from {original_value} to float: {value}")
                except Exception as e:
                    logger.warning(f"Error converting {col} to float (value: {value}): {e}, setting to 0.0")
                    value = 0.0
                    
            elif col in ['policy_csl', 'incident_state', 'incident_location', 'insured_hobbies', 'insured_relationship',
                         'insured_sex', 'insured_education_level', 'insured_occupation', 'policy_state']:
                try:
                    if value is None:
                        value = self.defaults.get(col, 'Unknown')
                        logger.debug(f"String field {col} was None, using default: {value}")
                    else:
                        value = str(value)
                        logger.debug(f"String field {col} set to: {value}")
                except Exception as e:
                    logger.warning(f"Error converting {col} to string: {e}")
                    value = 'Unknown'
            
            features[col] = value
        
        logger.info(f"Feature construction complete - {len(features)} features built")
        logger.info(f"Imputed {len(imputed_fields)} fields: {imputed_fields}")
        
        # Additional derived features (not in feature_columns but used later)
        logger.debug("Computing derived features")
        
        features['incident_in_policy_period'] = verification.get('incident_in_policy_period', False)
        logger.debug(f"incident_in_policy_period: {features['incident_in_policy_period']}")
        
        features['policy_status_at_incident'] = verification.get('policy_status', 'Active')
        logger.debug(f"policy_status_at_incident: {features['policy_status_at_incident']}")
        
        features['is_complex_claim'] = (
            features.get('total_claim_amount', 0) > 15000 or
            features.get('bodily_injuries', 0) > 1 or
            not features['incident_in_policy_period']
        )
        logger.info(f"is_complex_claim: {features['is_complex_claim']} - (amount>{15000}:{features.get('total_claim_amount',0)>15000}, injuries>1:{features.get('bodily_injuries',0)>1}, out_of_period:{not features['incident_in_policy_period']})")
        
        # Log summary statistics for key numeric features
        logger.debug(f"Key feature values - Risk score indicators:")
        logger.debug(f"  - credit_score: {features.get('credit_score')}")
        logger.debug(f"  - telematics_score: {features.get('telematics_score')}")
        logger.debug(f"  - prior_claims_count: {features.get('prior_claims_count')}")
        logger.debug(f"  - total_claim_amount: ${features.get('total_claim_amount', 0):,.2f}")
        logger.debug(f"  - policy_annual_premium: ${features.get('policy_annual_premium', 0):,.2f}")
        
        result = {
            'features': features,
            'feature_count': len(self.feature_columns),
            'imputed_count': len(imputed_fields),
            'imputed_fields': imputed_fields,
            'ready_for_ml': True
        }
        
        logger.info(f"Feature vector ready for ML - {result['feature_count']} features, {result['imputed_count']} imputed")
        return result
    
    def _compute_near_boundary(self, parsed: Dict, verification: Dict) -> int:
        """Return 1 if incident date is within 30 days of effective or expiration date, else 0"""
        logger.debug("Computing incident_near_boundary feature")
        
        try:
            incident_date_str = parsed.get('incident_date')
            eff_date_str = verification.get('effective_date')
            exp_date_str = verification.get('expiration_date')
            
            logger.debug(f"Dates - Incident: {incident_date_str}, Effective: {eff_date_str}, Expiration: {exp_date_str}")
            
            if not incident_date_str:
                logger.debug("No incident date provided, returning 0")
                return 0
                
            incident_date = datetime.strptime(incident_date_str, '%Y-%m-%d')
            logger.debug(f"Parsed incident date: {incident_date}")
            
            if eff_date_str:
                eff_date = datetime.strptime(eff_date_str, '%Y-%m-%d')
                days_from_start = (incident_date - eff_date).days
                logger.debug(f"Days from effective date: {days_from_start}")
                if 0 <= days_from_start <= 30:
                    logger.info(f"Incident within 30 days of policy start ({days_from_start} days) - boundary flagged")
                    return 1
                    
            if exp_date_str:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                days_to_expiry = (exp_date - incident_date).days
                logger.debug(f"Days to expiration: {days_to_expiry}")
                if 0 <= days_to_expiry <= 30:
                    logger.info(f"Incident within 30 days of policy expiration ({days_to_expiry} days) - boundary flagged")
                    return 1
                    
            logger.debug("Incident not near policy boundary (more than 30 days from start/end)")
            return 0
            
        except ValueError as e:
            logger.warning(f"Date parsing error in boundary calculation: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error in boundary calculation: {e}", exc_info=True)
            return 0