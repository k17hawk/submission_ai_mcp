# src/lang_graph/state.py
from typing import TypedDict, Optional, List, Any, Dict

class ClaimProcessingState(TypedDict):
    claim_text: str
    customer_name: str
    customer_id: str
    
    # All agent outputs as DICTIONARIES
    parsed_claim: Optional[Dict[str, Any]]
    policy_verification: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    feature_vector: Optional[Dict[str, Any]]
    fraud_prediction: Optional[Dict[str, Any]]
    policy_raw: Optional[Dict[str, Any]]
    
    final_decision: Optional[str]
    recommended_action: Optional[str]
    status: Optional[str]
    
    missing_fields: List[str]
    errors: List[str]
    processing_time_ms: float
    llm_enhanced: bool
    extraction_success: bool
    policy_found: bool
    critical_missing: List[str]