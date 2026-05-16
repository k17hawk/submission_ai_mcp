#!/usr/bin/env python3
"""
Main MCP Server - Central orchestrator for all insurance agents
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys

from src.mcp_submission_parsing.common.models import (
     ParsedClaim, PolicyVerification, RiskAssessment, 
    FeatureVector, FraudPrediction, CompleteClaimResponse, ClaimStatus
)

from src.mcp_submission_parsing.common.protocols import (
    Message, MessageType, AgentEndpoint, MCPServerProtocol
)

# Import agents
from mcp_submission_parsing.servers.agent_parser import ParserAgent
from src.mcp_submission_parsing.servers.agent_policy import PolicyAgent
from src.mcp_submission_parsing.servers.agent_risk import RiskAgent
from src.mcp_submission_parsing.servers.agent_features import FeatureAgent
from src.mcp_submission_parsing.servers.agent_fraud import FraudAgent


class MCPServer(MCPServerProtocol):
    """Main MCP Server that orchestrates all agents"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.agents = {}
        self.active_connections = {}
        self.request_history = []
        
        # Initialize agents
        self._init_agents()
    
    def _init_agents(self):
        """Initialize all agent handlers"""
        self.agents = {
            AgentEndpoint.PARSE_CLAIM: ParserAgent(),
            AgentEndpoint.LOOKUP_POLICY: PolicyAgent(),
            AgentEndpoint.CHECK_RISK: RiskAgent(),
            AgentEndpoint.BUILD_FEATURES: FeatureAgent(),
            AgentEndpoint.DETECT_FRAUD: FraudAgent(),
            AgentEndpoint.COMPLETE_PIPELINE: self._handle_complete_pipeline,
            AgentEndpoint.HEALTH_CHECK: self._handle_health_check,
        }
    
    async def start(self):
        """Start the MCP server"""
        print(f"🚀 Starting MCP Server on {self.host}:{self.port}")
        
        # Using stdio for simplicity (can be replaced with HTTP/WebSocket)
        print("📡 Server running in stdio mode")
        print("✅ Ready to accept requests")
        
        await self._stdio_loop()
    
    async def _stdio_loop(self):
        """Main loop for stdio communication"""
        while True:
            try:
                # Read message from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                # Parse message
                message = json.loads(line.strip())
                
                # Process request
                response = await self.handle_request(message)
                
                # Send response to stdout
                print(json.dumps(response))
                sys.stdout.flush()
                
            except Exception as e:
                error_response = {
                    "error": str(e),
                    "type": MessageType.ERROR.value
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming request and route to appropriate agent"""
        
        request_id = request.get('request_id', str(uuid.uuid4()))
        endpoint_str = request.get('endpoint')
        payload = request.get('payload', {})
        
        start_time = time.time()
        
        try:
            endpoint = AgentEndpoint(endpoint_str)
            
            if endpoint not in self.agents:
                return {
                    "request_id": request_id,
                    "error": f"Unknown endpoint: {endpoint}",
                    "type": MessageType.ERROR.value
                }
            
            # Call agent handler
            result = await self.agents[endpoint](payload)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Log request
            self.request_history.append({
                "request_id": request_id,
                "endpoint": endpoint_str,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "request_id": request_id,
                "result": result,
                "processing_time_ms": processing_time,
                "type": MessageType.RESPONSE.value
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "error": str(e),
                "type": MessageType.ERROR.value
            }
    
    async def _handle_complete_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate complete claim processing pipeline"""
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        claim_text = payload.get('claim_text', '')
        customer_name = payload.get('customer_name', '')
        customer_id = payload.get('customer_id', '')
        
        # Stage 1: Parse claim
        parse_result = await self.agents[AgentEndpoint.PARSE_CLAIM]({
            'text': claim_text,
            'use_llm': True
        })
        parsed_claim = ParsedClaim(**parse_result) if isinstance(parse_result, dict) else None
        
        if not parsed_claim or not parsed_claim.policy_number:
            return {
                "error": "Failed to extract policy number from claim",
                "status": "REJECTED"
            }
        
        # Stage 2: Lookup policy
        policy_result = await self.agents[AgentEndpoint.LOOKUP_POLICY]({
            'policy_number': parsed_claim.policy_number,
            'incident_date': parsed_claim.incident_date,
            'incident_type': parsed_claim.incident_type,
            'claim_amount': parsed_claim.total_claim_amount
        })
        policy_verification = PolicyVerification(**policy_result)
        
        if not policy_verification.found:
            return {
                "error": f"Policy {parsed_claim.policy_number} not found",
                "status": "REJECTED"
            }
        
        # Stage 3: Check risk rules
        risk_result = await self.agents[AgentEndpoint.CHECK_RISK]({
            'parsed': parsed_claim.to_dict(),
            'verification': policy_verification.to_dict()
        })
        risk_assessment = RiskAssessment(**risk_result)
        
        # Stage 4: Build features
        features_result = await self.agents[AgentEndpoint.BUILD_FEATURES]({
            'parsed': parsed_claim.to_dict(),
            'policy': policy_result.get('policy', {}),
            'verification': policy_verification.to_dict(),
            'customer_id': customer_id
        })
        feature_vector = FeatureVector(**features_result)
        
        # Stage 5: Detect fraud
        fraud_result = await self.agents[AgentEndpoint.DETECT_FRAUD]({
            'features': feature_vector.features,
            'claim_text': claim_text
        })
        fraud_prediction = FraudPrediction(**fraud_result)
        
        # Generate final decision
        final_decision, recommended_action = self._make_final_decision(
            risk_assessment, fraud_prediction
        )
        
        # Determine claim status
        if final_decision == "AUTO_APPROVE":
            status = ClaimStatus.APPROVED
        elif final_decision == "AUTO_DENY":
            status = ClaimStatus.DENIED
        elif risk_assessment.requires_siu:
            status = ClaimStatus.WITH_SIU
        else:
            status = ClaimStatus.UNDER_REVIEW
        
        # Generate summary
        summary = self._generate_summary(
            parsed_claim, policy_verification, risk_assessment, fraud_prediction
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        response = CompleteClaimResponse(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            status=status,
            parsed_claim=parsed_claim,
            policy_verification=policy_verification,
            risk_assessment=risk_assessment,
            fraud_prediction=fraud_prediction,
            final_decision=final_decision,
            recommended_action=recommended_action,
            summary=summary,
            processing_time_ms=processing_time
        )
        
        return response.to_dict()
    
    async def _handle_health_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents": list(self.agents.keys()),
            "active_connections": len(self.active_connections),
            "total_requests": len(self.request_history)
        }
    
    def _make_final_decision(self, risk: RiskAssessment, fraud: FraudPrediction) -> tuple:
        """Make final claim decision based on risk and fraud assessments"""
        
        if risk.risk_score >= 0.7 or fraud.fraud_probability >= 0.75:
            return "AUTO_DENY", "Refer to SIU with high priority"
        elif risk.risk_score >= 0.3 or fraud.fraud_probability >= 0.5:
            return "ADJUSTER_REVIEW", "Send to adjuster for detailed review"
        elif risk.risk_score <= 0.1 and fraud.fraud_probability <= 0.25:
            return "AUTO_APPROVE", "Auto-approve claim"
        else:
            return "MANUAL_REVIEW", "Manual review required"
    
    def _generate_summary(self, parsed: ParsedClaim, policy: PolicyVerification, 
                          risk: RiskAssessment, fraud: FraudPrediction) -> str:
        """Generate human-readable claim summary"""
        
        summary = f"""
CLAIM SUMMARY
=============
Policy: {parsed.policy_number}
Insured: {policy.customer_name}
Incident Date: {parsed.incident_date}
Incident Type: {parsed.incident_type}
Claim Amount: ${parsed.total_claim_amount:,.2f}

RISK ASSESSMENT
===============
Risk Score: {risk.risk_score:.3f} ({risk.risk_level})
Fraud Probability: {fraud.fraud_probability:.3f}
Violations: {len(risk.violations)}

DECISION
========
Final Decision: {self._make_final_decision(risk, fraud)[0]}
Recommended Action: {self._make_final_decision(risk, fraud)[1]}
"""
        return summary.strip()


async def main():
    """Main entry point"""
    server = MCPServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())