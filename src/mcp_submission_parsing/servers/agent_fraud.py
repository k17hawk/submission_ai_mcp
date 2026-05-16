"""Agent 5: Fraud Detection ML"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import sys


class FraudAgent:
    """Detects fraud using ML model"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.threshold = 0.5
        self.model_path = model_path or "/home/lang-chain/Documents/mcp_insurance/notebook/best_ensemble_fraud_model.pkl"
        self._load_model()
    
    def _load_model(self):
        """Load the trained ML model"""
        try:
            import joblib
            
            if Path(self.model_path).exists():
                payload = joblib.load(self.model_path)
                self.model = payload.get('pipeline')
                self.threshold = payload.get('threshold', 0.5)
                print(f"✅ Loaded fraud detection model (threshold: {self.threshold})")
            else:
                print(f"⚠️ Model not found at {self.model_path}, using rule-based fallback")
                self.model = None
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.model = None
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud probability"""
        
        features = payload.get('features', {})
        claim_text = payload.get('claim_text', '')
        
        # Use ML model if available
        if self.model:
            try:
                # Prepare DataFrame
                X = pd.DataFrame([features])
                
                # Get prediction
                proba = self.model.predict_proba(X)[0, 1]
                fraud_prob = float(proba)
                
            except Exception as e:
                print(f"ML prediction failed: {e}")
                fraud_prob = self._rule_based_fraud_score(features, claim_text)
        else:
            # Use rule-based fallback
            fraud_prob = self._rule_based_fraud_score(features, claim_text)
        
        # Determine flag and risk level
        fraud_flag = "Y" if fraud_prob >= self.threshold else "N"
        
        if fraud_prob >= 0.75:
            risk_level = "HIGH"
        elif fraud_prob >= self.threshold:
            risk_level = "MEDIUM"
        elif fraud_prob >= 0.25:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            'fraud_probability': round(fraud_prob, 4),
            'fraud_flag': fraud_flag,
            'risk_level': risk_level,
            'threshold_used': self.threshold,
            'requires_siu': fraud_prob >= 0.65,
            'model_version': '1.0'
        }
    
    def _rule_based_fraud_score(self, features: Dict[str, Any], claim_text: str) -> float:
        """Fallback rule-based fraud scoring"""
        score = 0.0
        
        # High claim amount
        if features.get('total_claim_amount', 0) > 25000:
            score += 0.3
        
        # Late night incident
        hour = features.get('incident_hour_of_the_day', 12)
        if hour >= 23 or hour <= 5:
            score += 0.2
        
        # No police report for large claim
        if features.get('total_claim_amount', 0) > 5000 and features.get('police_report_available') != 'YES':
            score += 0.25
        
        # Multiple prior claims
        if features.get('prior_claims_count', 0) > 2:
            score += 0.15
        
        # No witnesses
        if features.get('witnesses', 0) == 0 and features.get('number_of_vehicles_involved', 1) > 1:
            score += 0.1
        
        return min(score, 0.95)