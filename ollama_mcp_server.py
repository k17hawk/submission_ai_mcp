#!/usr/bin/env python3
import sys, time, asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_submission_parsing.common.models import (
    ParsedClaim, PolicyVerification, RiskAssessment,
    FeatureVector, FraudPrediction, CompleteClaimResponse, ClaimStatus
)
from src.mcp_submission_parsing.servers.agent_parser import ParserAgent
from src.mcp_submission_parsing.servers.agent_policy import PolicyAgent
from src.mcp_submission_parsing.servers.agent_risk import RiskAgent
from src.mcp_submission_parsing.servers.agent_features import FeatureAgent
from src.mcp_submission_parsing.servers.agent_fraud import FraudAgent

from fastmcp import FastMCP

import sys, numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class QuestionMarkToNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        return X.replace('?', np.nan)

import __main__
setattr(__main__, 'QuestionMarkToNaN', QuestionMarkToNaN)

mcp = FastMCP("insurance-claim-server")   # ← this is your server object

# Agent instances
parser_agent = ParserAgent()
policy_agent = PolicyAgent()
risk_agent = RiskAgent()
feature_agent = FeatureAgent()
fraud_agent = FraudAgent()

def _to_dict(result):
    return result if isinstance(result, dict) else result.to_dict()

# ── Tools ────────────────────────
@mcp.tool()
async def parse_claim(text: str, use_llm: bool = True) -> Dict[str, Any]:
    payload = {"text": text, "use_llm": use_llm}
    return _to_dict(await parser_agent(payload))

@mcp.tool()
async def lookup_policy(
    policy_number: str, incident_date: str,
    incident_type: str, claim_amount: float
) -> Dict[str, Any]:
    payload = {
        "policy_number": policy_number, "incident_date": incident_date,
        "incident_type": incident_type, "claim_amount": claim_amount
    }
    return _to_dict(await policy_agent(payload))

@mcp.tool()
async def check_risk(parsed: Dict[str, Any], verification: Dict[str, Any]) -> Dict[str, Any]:
    return _to_dict(await risk_agent({"parsed": parsed, "verification": verification}))

@mcp.tool()
async def build_features(
    parsed: Dict[str, Any], policy: Dict[str, Any],
    verification: Dict[str, Any], customer_id: str = ""
) -> Dict[str, Any]:
    payload = {
        "parsed": parsed, "policy": policy,
        "verification": verification, "customer_id": customer_id
    }
    return _to_dict(await feature_agent(payload))

@mcp.tool()
async def detect_fraud(features: Dict[str, Any], claim_text: str) -> Dict[str, Any]:
    return _to_dict(await fraud_agent({"features": features, "claim_text": claim_text}))

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": [
            "parse_claim", "lookup_policy", "check_risk",
            "build_features", "detect_fraud", "health_check",
            "complete_pipeline"
        ]
    }

@mcp.tool()
async def complete_pipeline(
    claim_text: str, customer_name: str = "", customer_id: str = ""
) -> Dict[str, Any]:
    """Complete claim processing pipeline - fails honestly on missing data"""
    start = time.time()
    
    # Step 1: Parse claim
    parse_result = await parser_agent({"text": claim_text, "use_llm": False})
    
    if parse_result.get('error'):
        return {
            "error": parse_result['error'],
            "status": "REJECTED",
            "final_decision": "SYSTEM_ERROR",
            "missing_data": parse_result.get('missing_fields', []),
            "extraction_confidence": parse_result.get('extraction_confidence', 0)
        }
    
    # Check if we have minimum required data
    required_for_processing = ['policy_number', 'incident_date', 'incident_type']
    missing = [f for f in required_for_processing if not parse_result.get(f)]
    
    if missing:
        return {
            "error": f"Cannot process claim: missing required fields: {', '.join(missing)}",
            "status": "REJECTED",
            "extracted_data": parse_result,
            "final_decision": "INCOMPLETE_SUBMISSION",
            "recommended_action": "Request missing information from customer"
        }
    
    # Step 2: Lookup policy with complete data
    policy_result = await policy_agent({
        "policy_number": parse_result['policy_number'],
        "incident_date": parse_result['incident_date'],
        "incident_type": parse_result['incident_type'],
        "claim_amount": parse_result.get('total_claim_amount', 0)
    })
    
    if not policy_result.get('found'):
        return {
            "error": f"Policy {parse_result['policy_number']} not found",
            "status": "REJECTED",
            "final_decision": "POLICY_NOT_FOUND",
            "extracted_data": parse_result
        }
    
    # Check for critical missing data
    critical_missing = []
    for field in ['deductible', 'csl', 'credit_score']:
        if policy_result.get(field) is None:
            critical_missing.append(field)
    
    if critical_missing:
        return {
            "error": f"Policy found but missing critical data: {', '.join(critical_missing)}",
            "status": "MANUAL_REVIEW_REQUIRED",
            "final_decision": "INCOMPLETE_POLICY_DATA",
            "policy_data": policy_result,
            "extracted_data": parse_result,
            "recommended_action": "Manual underwriting review required"
        }
    
    # Step 3-5: Continue processing only if we have complete data
    try:
        # Create ParsedClaim object
        parsed_claim = ParsedClaim(**parse_result)
        
        # Create PolicyVerification object
        policy_verification = PolicyVerification(**policy_result)
        
        # Risk assessment
        risk_result = await risk_agent({
            "parsed": parsed_claim.to_dict(),
            "verification": policy_verification.to_dict()
        })
        risk_assessment = RiskAssessment(**risk_result)
        
        # Feature building
        features_result = await feature_agent({
            "parsed": parsed_claim.to_dict(),
            "policy": policy_result,
            "verification": policy_verification.to_dict(),
            "customer_id": customer_id or policy_result.get('customer_id', '')
        })
        feature_vector = FeatureVector(**features_result)
        
        # Fraud detection
        fraud_result = await fraud_agent({
            "features": feature_vector.features,
            "claim_text": claim_text
        })
        fraud_prediction = FraudPrediction(**fraud_result)
        
        # Make decision
        risk_score = risk_assessment.risk_score
        fraud_prob = fraud_prediction.fraud_probability
        
        if risk_score >= 0.7 or fraud_prob >= 0.75:
            final_decision, recommended_action = "AUTO_DENY", "Refer to SIU with high priority"
        elif risk_score >= 0.3 or fraud_prob >= 0.5:
            final_decision, recommended_action = "ADJUSTER_REVIEW", "Send to adjuster for detailed review"
        elif risk_score <= 0.1 and fraud_prob <= 0.25:
            final_decision, recommended_action = "AUTO_APPROVE", "Auto-approve claim"
        else:
            final_decision, recommended_action = "MANUAL_REVIEW", "Manual review required"
        
        # Return complete response with warnings about missing data
        return {
            "status": "PROCESSED",
            "final_decision": final_decision,
            "recommended_action": recommended_action,
            "risk_score": risk_score,
            "fraud_probability": fraud_prob,
            "extracted_data": parse_result,
            "policy_data": {k: v for k, v in policy_result.items() if not callable(v)},
            "warnings": policy_result.get('warnings', []),
            "missing_data_warnings": critical_missing if critical_missing else None,
            "processing_time_ms": (time.time() - start) * 1000
        }
        
    except Exception as e:
        return {
            "error": f"Processing failed: {str(e)}",
            "status": "SYSTEM_ERROR",
            "final_decision": "MANUAL_REVIEW",
            "extracted_data": parse_result,
            "policy_data": policy_result,
            "processing_time_ms": (time.time() - start) * 1000
        }

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)