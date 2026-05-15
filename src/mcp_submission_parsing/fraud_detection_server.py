"""
MCP Server 5: Fraud Detection ML
Loads the trained ensemble model and scores incoming claims.
Includes derived 'calculated_total' column for model compatibility.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ────────────────────────────────────────────────────────────────────────────
# CUSTOM TRANSFORMER (must match the one used during training)
# ────────────────────────────────────────────────────────────────────────────
class QuestionMarkToNaN(BaseEstimator, TransformerMixin):
    """Replace '?' with NaN in the data."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X = X.replace('?', np.nan)
        return X


# ────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("/home/lang-chain/Documents/mcp_insurance/notebook/best_ensemble_fraud_model.pkl")

# ────────────────────────────────────────────────────────────────────────────
# IMPORTANT: These 39 columns MUST match the training data exactly.
# They are the base features used by the ColumnTransformer in the pipeline.
# ────────────────────────────────────────────────────────────────────────────
BASE_FEATURE_COLUMNS = [
    'months_as_customer',
    'age',
    'insured_sex',
    'insured_education_level',
    'insured_occupation',
    'insured_hobbies',
    'insured_relationship',
    'policy_state',
    'policy_csl',
    'policy_deductable',
    'policy_annual_premium',
    'umbrella_limit',
    'capital_gains',
    'capital_loss',
    'credit_score',
    'telematics_score',
    'incident_type',
    'collision_type',
    'incident_severity',
    'authorities_contacted',
    'incident_state',
    'incident_location',
    'incident_hour_of_the_day',
    'number_of_vehicles_involved',
    'property_damage',
    'bodily_injuries',
    'witnesses',
    'police_report_available',
    'total_claim_amount',
    'prior_claims_count',
    'auto_make',
    'auto_year',
    'injury_claim',
    'property_claim',
    'vehicle_claim',
    'incident_in_policy_period',
    'policy_status_at_incident',
    'incident_near_boundary',
    'is_complex_claim',
]

# Columns required by the trained model (base + derived)
MODEL_COLUMNS = BASE_FEATURE_COLUMNS + ['calculated_total']

_model = None
_threshold = 0.5


def _patch_main():
    """Inject QuestionMarkToNaN into __main__ so joblib can find it."""
    if 'QuestionMarkToNaN' not in sys.modules['__main__'].__dict__:
        sys.modules['__main__'].QuestionMarkToNaN = QuestionMarkToNaN
        logger.info("Patched __main__ with QuestionMarkToNaN.")


def load_model():
    """Load the saved pipeline and threshold from disk."""
    global _model, _threshold
    
    logger.info("=" * 60)
    logger.info("LOADING FRAUD DETECTION MODEL")
    logger.info("=" * 60)
    
    if _model is not None:
        logger.info("✅ Model already loaded in memory (cached).")
        logger.info(f"   Threshold: {_threshold:.4f}")
        return _model, _threshold
    
    logger.info(f"📂 Checking model path: {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        error_msg = f"Model file not found at {MODEL_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"✅ Model file found. Size: {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        logger.info("🔄 Patching __main__ for custom transformer...")
        _patch_main()
        
        logger.info("🔄 Loading model with joblib.load()...")
        payload = joblib.load(MODEL_PATH)
        logger.info("✅ Model payload loaded successfully.")
        
        _model = payload['pipeline']
        _threshold = payload.get('threshold', 0.5)
        
        logger.info(f"✅ Pipeline type: {type(_model).__name__}")
        logger.info(f"✅ Threshold: {_threshold:.4f}")
        
        if hasattr(_model, 'named_steps'):
            logger.info(f"📋 Pipeline steps: {list(_model.named_steps.keys())}")
        
        logger.info("=" * 60)
        logger.info("MODEL READY FOR PREDICTIONS")
        logger.info("=" * 60)
        
        return _model, _threshold
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.exception("Full traceback:")
        raise


def predict_fraud(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP Tool: Score a claim for fraud probability.
    
    Args:
        features: Dictionary containing the 39 base features from Agent 4.
                  Must include injury_claim, property_claim, vehicle_claim.
    
    Returns:
        Dictionary with fraud_probability, fraud_flag, risk_level, etc.
    """
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    logger.info("=" * 60)
    logger.info(f"🔍 PREDICTION REQUEST [{request_id}]")
    logger.info("=" * 60)
    
    try:
        model, threshold = load_model()
        logger.info(f"✅ Model loaded. Threshold: {threshold:.4f}")
        
        # Check for missing base features (39 expected)
        missing_base = [col for col in BASE_FEATURE_COLUMNS if col not in features]
        if missing_base:
            logger.warning(f"⚠️ Missing base features: {missing_base}")
        else:
            logger.info(f"✅ All {len(BASE_FEATURE_COLUMNS)} base features present.")
        
        # Build DataFrame with all base columns (fill missing with None)
        X_dict = {col: features.get(col) for col in BASE_FEATURE_COLUMNS}
        X = pd.DataFrame([X_dict])
        
        # Add derived column 'calculated_total'
        vehicle = features.get('vehicle_claim', 0)
        injury = features.get('injury_claim', 0)
        prop = features.get('property_claim', 0)
        X['calculated_total'] = vehicle + injury + prop
        
        # Reorder columns to match training order
        X = X[MODEL_COLUMNS]
        
        # Predict
        proba_array = model.predict_proba(X)
        proba = float(proba_array[0, 1])
        flag = "Y" if proba >= threshold else "N"
        
        # Risk categorization
        if proba >= 0.75:
            risk = "HIGH"
        elif proba >= threshold:
            risk = "MEDIUM"
        elif proba >= 0.25:
            risk = "LOW"
        else:
            risk = "MINIMAL"
        
        requires_siu = proba >= 0.65
        
        result = {
            "fraud_probability": round(proba, 4),
            "fraud_flag": flag,
            "risk_level": risk,
            "threshold_used": threshold,
            "requires_siu": requires_siu,
        }
        
        logger.info(f"📈 Fraud probability: {proba:.4f} -> {flag} (risk={risk})")
        return result
        
    except Exception as e:
        logger.error(f"❌ PREDICTION FAILED: {str(e)}")
        logger.exception("Full traceback:")
        return {
            "fraud_probability": -1.0,
            "fraud_flag": "ERROR",
            "risk_level": "UNKNOWN",
            "threshold_used": _threshold,
            "requires_siu": False,
        }