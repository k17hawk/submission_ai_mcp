"""Agent 4: Feature Builder for ML"""

from typing import Dict, Any
from datetime import datetime
from src.mcp_submission_parsing.common.transformer import QuestionMarkToNaN

class FeatureAgent:
    """Builds feature vector for fraud detection model"""
    
    def __init__(self):
        self.feature_columns = [
            'months_as_customer', 'age', 'insured_sex', 'insured_education_level',
            'insured_occupation', 'policy_state', 'policy_deductable', 
            'policy_annual_premium', 'credit_score', 'telematics_score',
            'incident_type', 'collision_type', 'incident_severity',
            'authorities_contacted', 'incident_hour_of_the_day',
            'number_of_vehicles_involved', 'property_damage', 'bodily_injuries',
            'witnesses', 'police_report_available', 'total_claim_amount',
            'prior_claims_count', 'auto_make', 'auto_year', 'injury_claim',
            'property_claim', 'vehicle_claim'
        ]
        
        self.defaults = {
            'credit_score': 650,
            'telematics_score': 70.0,
            'policy_deductable': 500,
            'policy_annual_premium': 1200.0,
            'prior_claims_count': 0,
            'bodily_injuries': 0,
            'witnesses': 0
        }
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build feature vector"""
        
        parsed = payload.get('parsed', {})
        policy = payload.get('policy', {})
        verification = payload.get('verification', {})
        customer_id = payload.get('customer_id', '')
        
        features = {}
        imputed_fields = []
        
        # Build features from available data
        for col in self.feature_columns:
            value = None
            
            # Try to get from parsed
            if col in parsed and parsed[col] is not None:
                value = parsed[col]
            # Try from policy
            elif col in policy and policy[col] is not None:
                value = policy[col]
            # Try from verification
            elif col in verification and verification[col] is not None:
                value = verification[col]
            # Use default
            else:
                value = self.defaults.get(col)
                if value is not None:
                    imputed_fields.append(col)
            
            # Type conversion
            if col in ['auto_year', 'bodily_injuries', 'witnesses', 
                       'prior_claims_count', 'incident_hour_of_the_day']:
                try:
                    value = int(value) if value else 0
                except:
                    value = 0
            elif col in ['total_claim_amount', 'injury_claim', 'property_claim', 
                         'vehicle_claim', 'policy_annual_premium', 'policy_deductable']:
                try:
                    value = float(value) if value else 0.0
                except:
                    value = 0.0
            
            features[col] = value
        
        # Calculate additional derived features
        features['incident_in_policy_period'] = verification.get('incident_in_policy_period', False)
        features['policy_status_at_incident'] = verification.get('policy_status', 'Active')
        features['is_complex_claim'] = (
            features.get('total_claim_amount', 0) > 15000 or
            features.get('bodily_injuries', 0) > 1 or
            not verification.get('incident_in_policy_period', True)
        )
        
        return {
            'features': features,
            'feature_count': len(self.feature_columns),
            'imputed_count': len(imputed_fields),
            'imputed_fields': imputed_fields,
            'ready_for_ml': True
        }