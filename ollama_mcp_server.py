#!/usr/bin/env python3
import sys, time, asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Agents (unchanged)
from src.mcp_submission_parsing.common.models import (
    ParsedClaim, PolicyVerification, RiskAssessment,
    FeatureVector, FraudPrediction, CompleteClaimResponse, ClaimStatus
)
from src.mcp_submission_parsing.servers.agent_parser import ParserAgent
from src.mcp_submission_parsing.servers.agent_policy import PolicyAgent
from src.mcp_submission_parsing.servers.agent_risk import RiskAgent
from src.mcp_submission_parsing.servers.agent_features import FeatureAgent
from src.mcp_submission_parsing.servers.agent_fraud import FraudAgent

# FastMCP imports (replaces the official mcp.server ones)
from fastmcp import FastMCP

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

# ── Complete pipeline ───────────
@mcp.tool()
async def complete_pipeline(
    claim_text: str, customer_name: str = "", customer_id: str = ""
) -> Dict[str, Any]:
    start = time.time()

    parse_result = await parser_agent({"text": claim_text, "use_llm": True})
    parsed_claim = ParsedClaim(**parse_result) if isinstance(parse_result, dict) else None
    if not parsed_claim or not parsed_claim.policy_number:
        return {"error": "Failed to extract policy number", "status": "REJECTED"}

    policy_result = await policy_agent({
        "policy_number": parsed_claim.policy_number,
        "incident_date": parsed_claim.incident_date,
        "incident_type": parsed_claim.incident_type,
        "claim_amount": parsed_claim.total_claim_amount
    })
    policy_verification = PolicyVerification(**policy_result)
    if not policy_verification.found:
        return {"error": f"Policy {parsed_claim.policy_number} not found", "status": "REJECTED"}

    risk_result = await risk_agent({
        "parsed": parsed_claim.to_dict(),
        "verification": policy_verification.to_dict()
    })
    risk_assessment = RiskAssessment(**risk_result)

    features_result = await feature_agent({
        "parsed": parsed_claim.to_dict(),
        "policy": policy_result.get("policy", {}),
        "verification": policy_verification.to_dict(),
        "customer_id": customer_id
    })
    feature_vector = FeatureVector(**features_result)

    fraud_result = await fraud_agent({
        "features": feature_vector.features,
        "claim_text": claim_text
    })
    fraud_prediction = FraudPrediction(**fraud_result)

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

    status_map = {
        "AUTO_APPROVE": ClaimStatus.APPROVED,
        "AUTO_DENY": ClaimStatus.DENIED,
    }
    status = status_map.get(final_decision)
    if status is None:
        status = ClaimStatus.WITH_SIU if risk_assessment.requires_siu else ClaimStatus.UNDER_REVIEW

    summary = f"""
CLAIM SUMMARY
=============
Policy: {parsed_claim.policy_number}
Insured: {policy_verification.customer_name}
Incident Date: {parsed_claim.incident_date}
Incident Type: {parsed_claim.incident_type}
Claim Amount: ${parsed_claim.total_claim_amount:,.2f}

RISK ASSESSMENT
===============
Risk Score: {risk_score:.3f} ({risk_assessment.risk_level})
Fraud Probability: {fraud_prob:.3f}
Violations: {len(risk_assessment.violations)}

DECISION
========
Final Decision: {final_decision}
Recommended Action: {recommended_action}
""".strip()

    response = CompleteClaimResponse(
        request_id="pipeline",
        timestamp=datetime.now().isoformat(),
        status=status,
        parsed_claim=parsed_claim,
        policy_verification=policy_verification,
        risk_assessment=risk_assessment,
        fraud_prediction=fraud_prediction,
        final_decision=final_decision,
        recommended_action=recommended_action,
        summary=summary,
        processing_time_ms=(time.time() - start) * 1000
    )
    return response.to_dict()

# ── Entry point ─────────────────
if __name__ == "__main__":
    mcp.run()   
    