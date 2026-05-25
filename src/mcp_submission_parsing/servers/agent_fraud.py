"""Agent 5: Fraud Detection ML"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from src.mcp_submission_parsing.config.logger_config import get_logger  
logger = get_logger('fraud agent')


import __main__
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class QuestionMarkToNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        return X.replace('?', np.nan)

__main__.QuestionMarkToNaN = QuestionMarkToNaN

class FraudAgent:
    """Detects fraud using ML model"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.threshold = 0.5
        self.model_path = model_path or "/home/lang-chain/Documents/mcp_insurance/notebook/best_ensemble_fraud_model.pkl"
        logger.info(f"Initializing FraudAgent with model path: {self.model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load the trained ML model"""
        try:
            import joblib
            
            if Path(self.model_path).exists():
                logger.info(f"Loading model from {self.model_path}")
                payload = joblib.load(self.model_path)
                self.model = payload.get('pipeline')
                self.threshold = payload.get('threshold', 0.5)
                logger.info(f"✅ Loaded fraud detection model (threshold: {self.threshold})")
            else:
                logger.warning(f"⚠️ Model not found at {self.model_path}, using rule-based fallback")
                self.model = None
        except Exception as e:
            logger.error(f"⚠️ Error loading model: {e}", exc_info=True)
            self.model = None
    
    async def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud probability"""
        
        features = payload.get('features', {})
        claim_text = payload.get('claim_text', '')
        logger.debug(f"Fraud detection called with claim_text length: {len(claim_text)}, features keys: {list(features.keys())}")
        
        if self.model:
            try:
                X = pd.DataFrame([features])
                logger.debug("Prepared feature DataFrame for ML prediction")
                
                proba = self.model.predict_proba(X)[0, 1]
                fraud_prob = float(proba)
                logger.info(f"ML prediction completed: fraud_probability = {fraud_prob:.4f}")
                
            except Exception as e:
                logger.error(f"ML prediction failed: {e}", exc_info=True)
                fraud_prob = self._rule_based_fraud_score(features, claim_text)
                logger.info(f"Fallback rule-based score used: {fraud_prob:.4f}")
        else:
            logger.info("No ML model available, using rule-based fallback")
            fraud_prob = self._rule_based_fraud_score(features, claim_text)
    
        fraud_flag = "Y" if fraud_prob >= self.threshold else "N"
        
        if fraud_prob >= 0.75:
            risk_level = "HIGH"
        elif fraud_prob >= self.threshold:
            risk_level = "MEDIUM"
        elif fraud_prob >= 0.25:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        requires_siu = fraud_prob >= 0.65
        logger.info(f"Fraud decision: probability={fraud_prob:.4f}, threshold={self.threshold}, flag={fraud_flag}, risk={risk_level}, SIU_required={requires_siu}")
        
        return {
            'fraud_probability': round(fraud_prob, 4),
            'fraud_flag': fraud_flag,
            'risk_level': risk_level,
            'threshold_used': self.threshold,
            'requires_siu': requires_siu,
            'model_version': '1.0'
        }
    
    def _rule_based_fraud_score(self, features: Dict[str, Any], claim_text: str) -> float:
        """Fallback rule-based fraud scoring"""
        logger.debug("Computing rule-based fraud score")
        score = 0.0
        
        total_claim = features.get('total_claim_amount', 0)
        if total_claim > 25000:
            score += 0.3
            logger.debug(f"High claim amount ${total_claim}: +0.3")
        
        hour = features.get('incident_hour_of_the_day', 12)
        if hour >= 23 or hour <= 5:
            score += 0.2
            logger.debug(f"Late night incident hour {hour}: +0.2")
        
        if total_claim > 5000 and features.get('police_report_available') != 'YES':
            score += 0.25
            logger.debug(f"Large claim ${total_claim} without police report: +0.25")
        
        prior_claims = features.get('prior_claims_count', 0)
        if prior_claims > 2:
            score += 0.15
            logger.debug(f"Multiple prior claims ({prior_claims}): +0.15")
        witnesses = features.get('witnesses', 0)
        vehicles = features.get('number_of_vehicles_involved', 1)
        if witnesses == 0 and vehicles > 1:
            score += 0.1
            logger.debug(f"No witnesses with {vehicles} vehicles involved: +0.1")
        
        final_score = min(score, 0.95)
        logger.debug(f"Rule-based score computed: {final_score:.4f}")
        return final_score