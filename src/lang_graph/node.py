# nodes.py
import time
from src.lang_graph.state import ClaimProcessingState
from src.mcp_submission_parsing.servers.agent_parser import ParserAgent
from src.mcp_submission_parsing.servers.agent_policy import PolicyAgent
from src.mcp_submission_parsing.servers.agent_risk import RiskAgent
from src.mcp_submission_parsing.servers.agent_features import FeatureAgent
from src.mcp_submission_parsing.servers.agent_fraud import FraudAgent
import json
# Lazy initialisation
_parser = None
_policy = None
_risk = None
_features = None
_fraud = None

def get_parser():
    global _parser
    if _parser is None:
        _parser = ParserAgent()
    return _parser

def get_policy():
    global _policy
    if _policy is None:
        _policy = PolicyAgent()
    return _policy

def get_risk():
    global _risk
    if _risk is None:
        _risk = RiskAgent()
    return _risk

def get_features():
    global _features
    if _features is None:
        _features = FeatureAgent()
    return _features

def get_fraud():
    global _fraud
    if _fraud is None:
        _fraud = FraudAgent()
    return _fraud

def _dict_to_parsed_claim(d: dict):
    from src.mcp_submission_parsing.common.models import ParsedClaim
    return ParsedClaim(**d) if d else None

def _dict_to_policy_verification(d: dict):
    from src.mcp_submission_parsing.common.models import PolicyVerification
    return PolicyVerification(**d) if d else None

def _dict_to_risk_assessment(d: dict):
    from src.mcp_submission_parsing.common.models import RiskAssessment
    return RiskAssessment(**d) if d else None

def _dict_to_feature_vector(d: dict):
    from src.mcp_submission_parsing.common.models import FeatureVector
    return FeatureVector(**d) if d else None

def _dict_to_fraud_prediction(d: dict):
    from src.mcp_submission_parsing.common.models import FraudPrediction
    return FraudPrediction(**d) if d else None

# ----------------------------------------------------------------------
# Nodes – store only dictionaries (msgpack‑friendly)
# ----------------------------------------------------------------------
async def claim_parsing_agent(state: ClaimProcessingState) -> dict:
    """Extract fields from claim text."""
    start = time.time()
    parser = get_parser()
    result = await parser({"text": state["claim_text"], "use_llm": False})
    
    missing = result.get("missing_fields", [])
    success = result.get("extraction_success", False)
    
    parsed_dict = None
    if not result.get("error"):
        from src.mcp_submission_parsing.common.models import ParsedClaim
        known_fields = {f.name for f in ParsedClaim.__dataclass_fields__.values()}
        filtered = {k: v for k, v in result.items() if k in known_fields}
        parsed_obj = ParsedClaim(**filtered)
        parsed_dict = parsed_obj.to_dict()   # ✅ store dict, not dataclass
    
    return {
        "parsed_claim": parsed_dict,
        "missing_fields": missing,
        "extraction_success": success,
        "processing_time_ms": (time.time() - start) * 1000,
        "llm_enhanced": result.get("llm_enhanced", False),
        "errors": result.get("error", []) if isinstance(result.get("error"), list) else [result.get("error")] if result.get("error") else []
    }

async def policy_lookup_agent(state: ClaimProcessingState) -> dict:
    """Lookup policy using extracted data."""
    start = time.time()
    parsed_dict = state["parsed_claim"]
    if not parsed_dict:
        return {"policy_found": False, "errors": ["No parsed claim data"]}
    
    # Reconstruct ParsedClaim from dict to access attributes
    parsed = _dict_to_parsed_claim(parsed_dict)
    if not parsed or not parsed.policy_number:
        return {"policy_found": False, "errors": ["No policy number extracted"]}
    
    policy_agent = get_policy()
    result = await policy_agent({
        "policy_number": parsed.policy_number,
        "incident_date": parsed.incident_date,
        "incident_type": parsed.incident_type,
        "claim_amount": parsed.total_claim_amount or 0,
        "auto_make": parsed.auto_make or ""
    })
    
    from src.mcp_submission_parsing.common.models import PolicyVerification
    known_fields = {f.name for f in PolicyVerification.__dataclass_fields__.values()}
    filtered = {k: v for k, v in result.items() if k in known_fields}
    policy_obj = PolicyVerification(**filtered)
    policy_dict = policy_obj.to_dict()   # ✅ store dict
    
    critical_fields = ['deductible', 'csl', 'credit_score']
    critical_missing = [f for f in critical_fields if result.get(f) is None]
    
    return {
        "policy_verification": policy_dict,
        "policy_raw": result,                     # already a dict
        "policy_found": result.get("found", False),
        "critical_missing": critical_missing,
        "processing_time_ms": (time.time() - start) * 1000
    }

async def risk_assessment_agent(state: ClaimProcessingState) -> dict:
    """Evaluate risk rules."""
    start = time.time()
    parsed_dict = state["parsed_claim"]
    policy_dict = state["policy_verification"]
    
    if not parsed_dict or not policy_dict:
        return {"errors": ["Missing parsed claim or policy data for risk assessment"]}
    
    risk_agent = get_risk()
    result = await risk_agent({
        "parsed": parsed_dict,       # dict already
        "verification": policy_dict  # dict already
    })
    
    from src.mcp_submission_parsing.common.models import RiskAssessment
    risk_obj = RiskAssessment(**result)
    risk_dict = risk_obj.to_dict()   # ✅ store dict
    
    return {
        "risk_assessment": risk_dict,
        "processing_time_ms": (time.time() - start) * 1000
    }

async def feature_building_agent(state: ClaimProcessingState) -> dict:
    """Build feature vector for ML."""
    start = time.time()
    parsed_dict = state["parsed_claim"]
    policy_dict = state["policy_verification"]
    
    if not parsed_dict or not policy_dict:
        return {"errors": ["Missing data for feature building"]}
    
    features_agent = get_features()
    policy_raw = state.get("policy_raw", {})
    
    result = await features_agent({
        "parsed": parsed_dict,
        "policy": policy_raw,
        "verification": policy_dict,
        "customer_id": state.get("customer_id", "")
    })
    
    from src.mcp_submission_parsing.common.models import FeatureVector
    feat_obj = FeatureVector(**result)
    feat_dict = feat_obj.to_dict()   

    feat_dict = json.loads(json.dumps(feat_dict, default=str))
    return {
        "feature_vector": feat_dict,
        "processing_time_ms": (time.time() - start) * 1000
    }

async def fraud_detection_agent(state: ClaimProcessingState) -> dict:
    """Run fraud detection ML."""
    start = time.time()
    feat_dict = state["feature_vector"]
    if not feat_dict or not feat_dict.get("features"):
        return {"errors": ["Missing feature vector for fraud detection"]}
    
    fraud_agent = get_fraud()
    result = await fraud_agent({
        "features": feat_dict["features"],  
        "claim_text": state["claim_text"]
    })
    
    from src.mcp_submission_parsing.common.models import FraudPrediction
    fraud_obj = FraudPrediction(**result)
    fraud_dict = fraud_obj.to_dict()   # ✅ store dict
    fraud_dict = json.loads(json.dumps(fraud_dict, default=str))
    return {
        "fraud_prediction": fraud_dict,
        "processing_time_ms": (time.time() - start) * 1000
    }

async def final_decision_agent(state: ClaimProcessingState) -> dict:
    """Combine risk and fraud scores to produce final decision."""
    risk_dict = state["risk_assessment"]
    fraud_dict = state["fraud_prediction"]
    
    if not risk_dict or not fraud_dict:
        return {"final_decision": "ERROR", "recommended_action": "System error – manual review required"}
    
    # Reconstruct to access attributes (or use dict keys directly)
    risk = _dict_to_risk_assessment(risk_dict)
    fraud = _dict_to_fraud_prediction(fraud_dict)
    
    risk_score = risk.risk_score if risk else 0
    fraud_prob = fraud.fraud_probability if fraud else 0
    
    if risk_score >= 0.70 or fraud_prob >= 0.75:
        final = "AUTO_DENY"
        action = "Refer to SIU with high priority"
        status = "DENIED"
    elif risk_score >= 0.30 or fraud_prob >= 0.50:
        final = "ADJUSTER_REVIEW"
        action = "Send to adjuster for detailed review"
        status = "UNDER_REVIEW"
    elif risk_score <= 0.10 and fraud_prob <= 0.25:
        final = "AUTO_APPROVE"
        action = "Auto-approve claim"
        status = "APPROVED"
    else:
        final = "MANUAL_REVIEW"
        action = "Manual review required"
        status = "UNDER_REVIEW"
    
    if not state.get("policy_found", False):
        final = "POLICY_NOT_FOUND"
        action = "Policy not found – contact customer"
        status = "DENIED"
    elif state.get("critical_missing"):
        final = "INCOMPLETE_POLICY_DATA"
        action = "Manual underwriting review required"
        status = "UNDER_REVIEW"
    
    return {
        "final_decision": final,
        "recommended_action": action,
        "status": status
    }

async def reject_claim_agent(state: ClaimProcessingState) -> dict:
    """Terminal node for rejected claims."""
    return {"final_decision": "REJECTED", "status": "REJECTED"}