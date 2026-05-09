"""
MCP Server 5: Fraud Detection ML
Loads the trained ensemble model and scores incoming claims.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

MODEL_PATH = Path("/home/lang-chain/Documents/mcp_insurance/notebook/best_ensemble_fraud_model.pkl")

FEATURE_COLUMNS = [
    'months_as_customer', 'age', 'insured_sex', 'insured_education_level',
    'insured_occupation', 'policy_deductable', 'umbrella_limit',
    'capital_gains', 'capital_loss', 'credit_score', 'telematics_score',
    'incident_type', 'collision_type', 'incident_severity',
    'authorities_contacted', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'property_damage', 'bodily_injuries',
    'witnesses', 'police_report_available', 'total_claim_amount',
    'prior_claims_count', 'incident_in_policy_period',
    'policy_status_at_incident', 'incident_near_boundary',
    'is_complex_claim',
]

_model = None
_threshold = 0.5

def load_model():
    global _model, _threshold
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        payload = joblib.load(MODEL_PATH)
        _model = payload['pipeline']
        _threshold = payload.get('threshold', 0.5)
        print(f"✅ Fraud model loaded. Threshold: {_threshold:.4f}")
    return _model, _threshold


def predict_fraud(features: Dict[str, Any]) -> Dict[str, Any]:
    """MCP Tool: Score a claim for fraud probability. Guarantees no None."""
    try:
        model, threshold = load_model()
        X = pd.DataFrame([features])[FEATURE_COLUMNS]
        proba = float(model.predict_proba(X)[0, 1])
        flag = "Y" if proba >= threshold else "N"
        
        if proba >= 0.75:
            risk = "HIGH"
        elif proba >= threshold:
            risk = "MEDIUM"
        elif proba >= 0.25:
            risk = "LOW"
        else:
            risk = "MINIMAL"
        
        return {
            "fraud_probability": round(proba, 4),
            "fraud_flag": flag,
            "risk_level": risk,
            "threshold_used": threshold,
            "requires_siu": proba >= 0.65,
        }
    except Exception as e:
        return {
            "fraud_probability": -1.0,
            "fraud_flag": "ERROR",
            "risk_level": "UNKNOWN",
            "threshold_used": _threshold,
            "requires_siu": False,
            "error": str(e),
        }