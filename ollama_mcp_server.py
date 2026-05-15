#!/usr/bin/env python3
"""
MCP Orchestrator - Central host for all insurance claim processing servers
Integrates all 5 agents with Ollama LLM for intelligent decision making
"""

import asyncio
import json
import httpx
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import your existing servers
from src.mcp_submission_parsing.parser_server import SubmissionParser
from src.mcp_submission_parsing.policy_lookup_server import (
    PolicyLookupService, lookup_policy, verify_identity, 
    check_coverage, pre_fill_claim, INCIDENT_TO_COVERAGE
)
from src.mcp_submission_parsing.rule_checker_server import check_risk_rules
from src.mcp_submission_parsing.feature_builder_server import build_feature_vector
from src.mcp_submission_parsing.fraud_detection_server import predict_fraud

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"
DATA_PATH = "/home/lang-chain/Documents/mcp_insurance/underwriting_50k_dataset.xlsx"

# Initialize services
parser = SubmissionParser()
policy_service = PolicyLookupService(DATA_PATH)

# Create MCP server
mcp = FastMCP("insurance-claims-orchestrator")

# ============================================================================
# OLLAMA INTEGRATION TOOLS
# ============================================================================

async def call_ollama(prompt: str, temperature: float = 0.0, system_prompt: str = None) -> str:
    """Helper to call Ollama with consistent configuration"""
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 2000
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()


@mcp.tool()
async def llm_analyze_claim_complexity(claim_text: str, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLM to analyze claim complexity and suggest handling priority.
    """
    prompt = f"""
Analyze this insurance claim and determine:
1. Complexity level (LOW/MEDIUM/HIGH)
2. Priority (ROUTINE/PRIORITY/URGENT)
3. Recommended handling path (AUTO_PROCESS/ADJUSTER_REVIEW/SIU_REVIEW)
4. Key risk indicators identified
5. Suggested next steps

Claim Text: {claim_text[:2000]}

Parsed Data: {json.dumps(parsed_data, indent=2)}

Respond in JSON format with these keys:
- complexity
- priority
- recommended_handling
- risk_indicators (list)
- next_steps (list)
"""
    
    response = await call_ollama(prompt, temperature=0.2)
    try:
        return json.loads(response)
    except:
        return {
            "complexity": "MEDIUM",
            "priority": "ROUTINE",
            "recommended_handling": "ADJUSTER_REVIEW",
            "risk_indicators": ["Unable to parse LLM response"],
            "next_steps": ["Manual review required"],
            "raw_response": response
        }


@mcp.tool()
async def llm_extract_vehicle_details(text: str) -> Dict[str, Any]:
    """
    Use LLM to extract vehicle information with high accuracy.
    """
    prompt = f"""
Extract vehicle details from the following text.
Return ONLY valid JSON with these fields:
- year (4-digit number, null if not found)
- make (string, null if not found)
- model (string, null if not found)
- vin (17-character VIN, null if not found)
- confidence (score 0-1)

Text: {text}

JSON:
"""
    response = await call_ollama(prompt, temperature=0.1)
    try:
        return json.loads(response)
    except:
        return {"year": None, "make": None, "model": None, "vin": None, "confidence": 0.0}


@mcp.tool()
async def llm_assess_claim_veracity(claim_text: str, verification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLM to assess potential fraud indicators and claim veracity.
    """
    prompt = f"""
As a fraud detection expert, assess this insurance claim for potential fraud indicators.

Claim Text: {claim_text[:2000]}

Verification Results: {json.dumps(verification, indent=2)}

Analyze for:
1. Inconsistencies in the narrative
2. Suspicious patterns or timing
3. Red flags in the description
4. Overall veracity score (0-100)
5. Top 3 concerns (if any)

Return JSON with keys:
- veracity_score (0-100)
- suspicious_elements (list of strings)
- fraud_risk_level (LOW/MEDIUM/HIGH)
- concerns (list of strings)
- recommended_action
"""
    
    response = await call_ollama(prompt, temperature=0.3)
    try:
        return json.loads(response)
    except:
        return {
            "veracity_score": 50,
            "suspicious_elements": [],
            "fraud_risk_level": "MEDIUM",
            "concerns": ["Unable to analyze with LLM"],
            "recommended_action": "Manual review required"
        }


@mcp.tool()
async def llm_generate_claim_summary(
    claim_text: str, 
    parsed_data: Dict[str, Any], 
    risk_assessment: Dict[str, Any]
) -> str:
    """
    Generate a comprehensive claim summary for adjusters.
    """
    prompt = f"""
Generate a concise, professional insurance claim summary for an adjuster.

Include:
- Incident overview (what, when, where)
- Policy holder info
- Vehicle information
- Damage assessment
- Risk findings
- Recommended action

Claim Text: {claim_text}

Parsed Data: {json.dumps(parsed_data, indent=2)}

Risk Assessment: {json.dumps(risk_assessment, indent=2)}

Summary:
"""
    
    return await call_ollama(prompt, temperature=0.4, system_prompt="You are an insurance claims adjuster writing a professional summary.")


# ============================================================================
# AGENT 1: SUBMISSION PARSER TOOLS (with LLM enhancement)
# ============================================================================

@mcp.tool()
async def parse_claim_submission(text: str, doc_type: str = "unknown", use_llm: bool = True) -> Dict[str, Any]:
    """
    Parse claim submission text using regex + spaCy, optionally enhanced with LLM.
    """
    # First use deterministic extraction
    result = parser.parse(text, doc_type)
    parsed = result.__dict__
    
    # Enhance with LLM if requested
    if use_llm and text:
        try:
            llm_vehicle = await llm_extract_vehicle_details(text)
            if llm_vehicle.get("year") and not parsed.get("auto_year"):
                parsed["auto_year"] = llm_vehicle["year"]
            if llm_vehicle.get("make") and not parsed.get("auto_make"):
                parsed["auto_make"] = llm_vehicle["make"]
            if llm_vehicle.get("model") and not parsed.get("auto_model"):
                parsed["auto_model"] = llm_vehicle["model"]
            parsed["llm_enhanced"] = True
        except Exception as e:
            parsed["llm_enhanced"] = False
            parsed["llm_error"] = str(e)
    
    return parsed


@mcp.tool()
async def parse_multiple_documents(documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """Parse multiple documents and merge results."""
    result = parser.parse_multiple(documents)
    return result.__dict__


@mcp.tool()
async def validate_extraction(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extracted fields."""
    import re
    errors = []
    if parsed.get('policy_number') and not re.match(r'^POL-\d{6}$', str(parsed['policy_number'])):
        errors.append(f"Invalid policy_number format: {parsed['policy_number']}")
    if parsed.get('auto_year') and not (1980 <= int(parsed['auto_year']) <= 2025):
        errors.append(f"auto_year out of range: {parsed['auto_year']}")
    if parsed.get('total_claim_amount') and float(parsed['total_claim_amount']) <= 0:
        errors.append("total_claim_amount must be positive")
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': [],
        'completeness': 1.0 - (len(parsed.get('missing_fields', [])) / 4),
    }


# ============================================================================
# AGENT 2: POLICY LOOKUP TOOLS
# ============================================================================

@mcp.tool()
async def lookup_policy_info(policy_number: str) -> Dict[str, Any]:
    """Lookup policy information by policy number."""
    return lookup_policy(policy_number)


@mcp.tool()
async def verify_claimant_identity(
    customer_name: str = "", 
    customer_id: str = "", 
    policy_number: str = ""
) -> Dict[str, Any]:
    """Verify claimant identity against policy records."""
    return verify_identity(customer_name, customer_id, policy_number)


@mcp.tool()
async def check_coverage_eligibility(
    policy_number: str,
    incident_type: str,
    incident_date: str,
    claim_amount: float = 0.0
) -> Dict[str, Any]:
    """Check if incident is covered under the policy."""
    return check_coverage(policy_number, incident_type, incident_date, claim_amount)


@mcp.tool()
async def prefill_claim_form(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-fill claim form with policy data."""
    return pre_fill_claim(parsed_data)


# ============================================================================
# AGENT 3: RISK RULE CHECKER TOOLS
# ============================================================================

@mcp.tool()
async def evaluate_risk_rules(
    parsed: Dict[str, Any],
    verification: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate all risk rules against claim data."""
    return check_risk_rules(parsed, verification)


# ============================================================================
# AGENT 4: FEATURE BUILDER TOOLS
# ============================================================================

@mcp.tool()
async def build_ml_features(
    parsed: Dict[str, Any],
    policy: Dict[str, Any],
    verification: Dict[str, Any],
    customer_profile: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Build feature vector for fraud detection ML model."""
    return build_feature_vector(parsed, policy, verification, customer_profile)


# ============================================================================
# AGENT 5: FRAUD DETECTION TOOLS
# ============================================================================

@mcp.tool()
async def detect_fraud(features: Dict[str, Any]) -> Dict[str, Any]:
    """Score claim for fraud probability using ML model."""
    return predict_fraud(features)


# ============================================================================
# ORCHESTRATION TOOL - Complete Pipeline
# ============================================================================

@mcp.tool()
async def process_complete_claim(
    claim_text: str,
    customer_name: str = "",
    customer_id: str = ""
) -> Dict[str, Any]:
    """
    End-to-end claim processing through all 5 agents.
    """
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "stages": {},
        "final_decision": None,
        "recommended_action": None
    }
    
    # Stage 1: Parse submission
    try:
        parsed = await parse_claim_submission(claim_text, "claim_form", use_llm=True)
        result["stages"]["parser"] = {
            "status": "success",
            "data": parsed,
            "completeness": parsed.get("missing_fields", [])
        }
    except Exception as e:
        result["stages"]["parser"] = {"status": "failed", "error": str(e)}
        result["final_decision"] = "ERROR"
        return result
    
    # Stage 2: Policy lookup and verification
    try:
        policy_number = parsed.get("policy_number")
        if not policy_number:
            result["stages"]["policy"] = {"status": "failed", "error": "No policy number extracted"}
            result["final_decision"] = "REJECTED"
            return result
        
        policy_info = await lookup_policy_info(policy_number)
        if not policy_info.get("found"):
            result["stages"]["policy"] = {"status": "failed", "error": "Policy not found"}
            result["final_decision"] = "REJECTED"
            return result
        
        # Verify identity
        identity = await verify_claimant_identity(
            customer_name=customer_name or parsed.get("insured_name", ""),
            customer_id=customer_id,
            policy_number=policy_number
        )
        
        # Check coverage
        coverage = await check_coverage_eligibility(
            policy_number,
            parsed.get("incident_type", ""),
            parsed.get("incident_date", ""),
            parsed.get("total_claim_amount", 0)
        )
        
        # Pre-fill
        prefill = await prefill_claim_form(parsed)
        
        result["stages"]["policy"] = {
            "status": "success",
            "policy_info": policy_info.get("policy", {}),
            "identity_verified": identity.get("verified", False),
            "coverage_eligible": coverage.get("covered", False),
            "prefill_data": prefill
        }
        
        verification_data = prefill.get("verification", {})
        
    except Exception as e:
        result["stages"]["policy"] = {"status": "failed", "error": str(e)}
        result["final_decision"] = "ERROR"
        return result
    
    # Stage 3: Risk assessment
    try:
        risk_assessment = await evaluate_risk_rules(parsed, verification_data)
        result["stages"]["risk"] = {
            "status": "success",
            "risk_score": risk_assessment.get("risk_score"),
            "risk_level": risk_assessment.get("risk_level"),
            "violations": risk_assessment.get("violations", []),
            "auto_decision": risk_assessment.get("auto_decision")
        }
    except Exception as e:
        result["stages"]["risk"] = {"status": "failed", "error": str(e)}
    
    # Stage 4: Build ML features
    try:
        customer_profile = None
        if customer_id:
            customer_profile = policy_service.get_customer(customer_id)
        
        features = await build_ml_features(
            parsed,
            policy_info.get("policy", {}),
            verification_data,
            customer_profile
        )
        result["stages"]["features"] = {
            "status": "success",
            "feature_count": features.get("feature_count"),
            "imputed_count": features.get("imputed_count"),
            "ready_for_ml": features.get("ready_for_ml")
        }
    except Exception as e:
        result["stages"]["features"] = {"status": "failed", "error": str(e)}
        features = {"features": {}}
    
    # Stage 5: Fraud detection
    try:
        fraud_result = await detect_fraud(features.get("features", {}))
        result["stages"]["fraud"] = {
            "status": "success",
            "fraud_probability": fraud_result.get("fraud_probability"),
            "fraud_flag": fraud_result.get("fraud_flag"),
            "risk_level": fraud_result.get("risk_level")
        }
    except Exception as e:
        result["stages"]["fraud"] = {"status": "failed", "error": str(e)}
    
    # Stage 6: LLM final assessment
    try:
        llm_assessment = await llm_analyze_claim_complexity(claim_text, parsed)
        llm_veracity = await llm_assess_claim_veracity(claim_text, verification_data)
        
        result["stages"]["llm"] = {
            "status": "success",
            "complexity": llm_assessment.get("complexity"),
            "recommended_handling": llm_assessment.get("recommended_handling"),
            "veracity_score": llm_veracity.get("veracity_score"),
            "fraud_risk_level": llm_veracity.get("fraud_risk_level")
        }
        
        # Generate final decision
        risk_score = risk_assessment.get("risk_score", 0)
        fraud_prob = fraud_result.get("fraud_probability", 0)
        
        if risk_score >= 0.7 or fraud_prob >= 0.75:
            result["final_decision"] = "AUTO_DENY"
            result["recommended_action"] = "Refer to SIU with high priority"
        elif risk_score >= 0.3 or fraud_prob >= 0.5:
            result["final_decision"] = "ADJUSTER_REVIEW"
            result["recommended_action"] = "Send to adjuster for detailed review"
        elif llm_veracity.get("fraud_risk_level") == "HIGH":
            result["final_decision"] = "SIU_REVIEW"
            result["recommended_action"] = "Flag for Special Investigation Unit"
        else:
            result["final_decision"] = "AUTO_APPROVE"
            result["recommended_action"] = "Auto-approve claim"
            
    except Exception as e:
        result["stages"]["llm"] = {"status": "failed", "error": str(e)}
        result["final_decision"] = "MANUAL_REVIEW"
        result["recommended_action"] = "Manual review required due to LLM error"
    
    # Generate summary
    try:
        summary = await llm_generate_claim_summary(claim_text, parsed, risk_assessment)
        result["summary"] = summary
    except:
        result["summary"] = "Summary generation failed"
    
    return result


# ============================================================================
# ADDITIONAL UTILITY TOOLS
# ============================================================================

@mcp.tool()
async def get_system_status() -> Dict[str, Any]:
    """Get status of all connected services."""
    status = {
        "ollama": {"status": "checking", "model": MODEL_NAME},
        "policy_service": {"status": "active", "policies_loaded": len(policy_service._policies)},
        "parser": {"status": "active"},
        "risk_engine": {"status": "active"},
        "fraud_model": {"status": "pending"}
    }
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                status["ollama"]["status"] = "connected"
            else:
                status["ollama"]["status"] = "error"
    except:
        status["ollama"]["status"] = "disconnected"
    
    # Check fraud model
    try:
        from src.mcp_submission_parsing.fraud_detection_server import load_model
        model, _ = load_model()
        status["fraud_model"]["status"] = "loaded" if model else "failed"
    except:
        status["fraud_model"]["status"] = "not_loaded"
    
    return status


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Insurance Claims MCP Orchestrator with Ollama LLM")
    print("=" * 60)
    print(f"Ollama Host: {OLLAMA_HOST}")
    print(f"Model: {MODEL_NAME}")
    print(f"Data Path: {DATA_PATH}")
    print("=" * 60)
    print("Available tools:")
    print("  - parse_claim_submission")
    print("  - lookup_policy_info")
    print("  - verify_claimant_identity")
    print("  - check_coverage_eligibility")
    print("  - evaluate_risk_rules")
    print("  - build_ml_features")
    print("  - detect_fraud")
    print("  - process_complete_claim (end-to-end)")
    print("  - llm_analyze_claim_complexity")
    print("  - llm_extract_vehicle_details")
    print("  - llm_assess_claim_veracity")
    print("  - llm_generate_claim_summary")
    print("=" * 60)
    print("\nStarting MCP server...")
    print("\nTo test: python -m mcp.client.cli run mcp_orchestrator.py")
    print("\nOr use with Claude Desktop:")
    print("Add to claude_desktop_config.json:")
    
    print("=" * 60)
    
    mcp.run()
    