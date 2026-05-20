#!/usr/bin/env python3
"""
Ollama MCP Server - Insurance Claim Processing
Lazy loading of agents to avoid memory exhaustion at startup.
"""

import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_submission_parsing.common.models import (
    ParsedClaim, PolicyVerification, RiskAssessment,
    FeatureVector, FraudPrediction, ClaimStatus
)
from src.mcp_submission_parsing.servers.agent_parser import ParserAgent
from src.mcp_submission_parsing.servers.agent_policy import PolicyAgent
from src.mcp_submission_parsing.servers.agent_risk import RiskAgent
from src.mcp_submission_parsing.servers.agent_features import FeatureAgent
from src.mcp_submission_parsing.servers.agent_fraud import FraudAgent

from fastmcp import FastMCP
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------
# Required for pickle deserialization of the fraud model
class QuestionMarkToNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        return X.replace('?', np.nan)

# Make it available for unpickling
import __main__
__main__.QuestionMarkToNaN = QuestionMarkToNaN
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Lazy agent wrapper
class LazyAgent:
    """Creates the real agent only when first accessed."""
    def __init__(self, creator):
        self.creator = creator
        self._instance = None

    def __call__(self):
        if self._instance is None:
            self._instance = self.creator()
        return self._instance

def _create_parser() -> ParserAgent:
    return ParserAgent()

def _create_policy() -> PolicyAgent:
    return PolicyAgent()

def _create_risk() -> RiskAgent:
    return RiskAgent()

def _create_features() -> FeatureAgent:
    return FeatureAgent()

def _create_fraud() -> FraudAgent:
    return FraudAgent()

# Lazy agent instances (no heavy loading at startup)
parser_agent = LazyAgent(_create_parser)
policy_agent = LazyAgent(_create_policy)
risk_agent = LazyAgent(_create_risk)
feature_agent = LazyAgent(_create_features)
fraud_agent = LazyAgent(_create_fraud)
# ---------------------------------------------------------------------

mcp = FastMCP("insurance-claim-server")

def _to_dict(result):
    return result if isinstance(result, dict) else result.to_dict()

# ---------------------------------------------------------------------
# Tools (each tool resolves the lazy agent on first call)
# ---------------------------------------------------------------------
@mcp.tool()
async def parse_claim(text: str, use_llm: bool = True) -> Dict[str, Any]:
    agent = parser_agent()
    return _to_dict(await agent({"text": text, "use_llm": use_llm}))

@mcp.tool()
async def lookup_policy(
    policy_number: str,
    incident_date: str,
    incident_type: str,
    claim_amount: float,
    auto_make: str = ""
) -> Dict[str, Any]:
    agent = policy_agent()
    payload = {
        "policy_number": policy_number,
        "incident_date": incident_date,
        "incident_type": incident_type,
        "claim_amount": claim_amount,
        "auto_make": auto_make
    }
    return _to_dict(await agent(payload))

@mcp.tool()
async def check_risk(parsed: Dict[str, Any], verification: Dict[str, Any]) -> Dict[str, Any]:
    agent = risk_agent()
    return _to_dict(await agent({"parsed": parsed, "verification": verification}))

@mcp.tool()
async def build_features(
    parsed: Dict[str, Any],
    policy: Dict[str, Any],
    verification: Dict[str, Any],
    customer_id: str = ""
) -> Dict[str, Any]:
    agent = feature_agent()
    payload = {
        "parsed": parsed,
        "policy": policy,
        "verification": verification,
        "customer_id": customer_id
    }
    return _to_dict(await agent(payload))

@mcp.tool()
async def detect_fraud(features: Dict[str, Any], claim_text: str) -> Dict[str, Any]:
    agent = fraud_agent()
    return _to_dict(await agent({"features": features, "claim_text": claim_text}))

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
    claim_text: str,
    customer_name: str = "",
    customer_id: str = ""
) -> Dict[str, Any]:
    """
    Complete claim processing pipeline – uses full risk rules (including policy period)
    """
    import dataclasses
    start = time.time()

    # Step 1: Parse claim (no LLM forced initially)
    parser = parser_agent()
    parse_result = await parser({"text": claim_text, "use_llm": False})

    if parse_result.get('error'):
        return {
            "error": parse_result['error'],
            "status": "REJECTED",
            "final_decision": "SYSTEM_ERROR",
            "missing_data": parse_result.get('missing_fields', []),
            "extraction_confidence": parse_result.get('extraction_confidence', 0)
        }

    # Check minimum required data
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

    # Step 2: Lookup policy (includes warnings for make mismatch, low scores, etc.)
    policy_agent_instance = policy_agent()
    policy_result = await policy_agent_instance({
        "policy_number": parse_result['policy_number'],
        "incident_date": parse_result['incident_date'],
        "incident_type": parse_result['incident_type'],
        "claim_amount": parse_result.get('total_claim_amount', 0),
        "auto_make": parse_result.get('auto_make', '')
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

    # Step 3: Risk assessment (uses risk_rules.yaml – includes R002 policy period)
    known_parsed_fields = {f.name for f in dataclasses.fields(ParsedClaim)}
    filtered_parse = {k: v for k, v in parse_result.items() if k in known_parsed_fields}
    parsed_claim = ParsedClaim(**filtered_parse)

    known_policy_fields = {f.name for f in dataclasses.fields(PolicyVerification)}
    filtered_policy = {k: v for k, v in policy_result.items() if k in known_policy_fields}
    policy_verification = PolicyVerification(**filtered_policy)

    risk_agent_instance = risk_agent()
    risk_result = await risk_agent_instance({
        "parsed": parsed_claim.to_dict(),
        "verification": policy_verification.to_dict()
    })
    risk_assessment = RiskAssessment(**risk_result)

    # Step 4: Build features (only if policy data is complete enough)
    feature_agent_instance = feature_agent()
    features_result = await feature_agent_instance({
        "parsed": parsed_claim.to_dict(),
        "policy": policy_result,
        "verification": policy_verification.to_dict(),
        "customer_id": customer_id or policy_result.get('customer_id', '')
    })
    known_feature_fields = {f.name for f in dataclasses.fields(FeatureVector)}
    filtered_features = {k: v for k, v in features_result.items() if k in known_feature_fields}
    feature_vector = FeatureVector(**filtered_features)

    # Step 5: Fraud detection
    fraud_agent_instance = fraud_agent()
    fraud_result = await fraud_agent_instance({
        "features": feature_vector.features,
        "claim_text": claim_text
    })
    known_fraud_fields = {f.name for f in dataclasses.fields(FraudPrediction)}
    filtered_fraud = {k: v for k, v in fraud_result.items() if k in known_fraud_fields}
    fraud_prediction = FraudPrediction(**filtered_fraud)

    # Step 6: Final decision based on risk assessment (which includes all rules)
    risk_score = risk_assessment.risk_score
    fraud_prob = fraud_prediction.fraud_probability

    # Auto‑decision rules (aligned with risk rules thresholds)
    if risk_score >= 0.70 or fraud_prob >= 0.75:
        final_decision = "AUTO_DENY"
        recommended_action = "Refer to SIU with high priority"
        status = ClaimStatus.DENIED
    elif risk_score >= 0.30 or fraud_prob >= 0.50:
        final_decision = "ADJUSTER_REVIEW"
        recommended_action = "Send to adjuster for detailed review"
        status = ClaimStatus.UNDER_REVIEW
    elif risk_score <= 0.10 and fraud_prob <= 0.25:
        final_decision = "AUTO_APPROVE"
        recommended_action = "Auto-approve claim"
        status = ClaimStatus.APPROVED
    else:
        final_decision = "MANUAL_REVIEW"
        recommended_action = "Manual review required"
        status = ClaimStatus.UNDER_REVIEW

    # Override for policy not found or critical missing data
    if not policy_result.get('found'):
        final_decision = "POLICY_NOT_FOUND"
        status = ClaimStatus.DENIED
    elif critical_missing:
        final_decision = "INCOMPLETE_POLICY_DATA"
        recommended_action = "Manual underwriting review required"
        status = ClaimStatus.UNDER_REVIEW

    # Prepare response
    response = {
        "status": status.value if hasattr(status, 'value') else str(status),
        "final_decision": final_decision,
        "recommended_action": recommended_action,
        "risk_score": risk_score,
        "fraud_probability": fraud_prob,
        "extracted_data": parse_result,
        "policy_data": {k: v for k, v in policy_result.items() if not callable(v)},
        "risk_violations": risk_assessment.violations,
        "risk_warnings": risk_assessment.warnings,
        "policy_warnings": policy_result.get('warnings', []),
        "missing_data_warnings": critical_missing if critical_missing else None,
        "processing_time_ms": (time.time() - start) * 1000
    }

    if risk_assessment.requires_siu:
        response["siu_referral"] = True
        response["recommended_action"] = "Immediate SIU referral"

    return response

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)