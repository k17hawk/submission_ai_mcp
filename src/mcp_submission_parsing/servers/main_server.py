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
from src.mcp_submission_parsing.config.logger_config import get_logger  
logger = get_logger('main agent')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_submission_parsing.common.models import (
     ParsedClaim, PolicyVerification, RiskAssessment, 
    FeatureVector, FraudPrediction, CompleteClaimResponse, ClaimStatus
)

from src.mcp_submission_parsing.common.protocols import (
    Message, MessageType, AgentEndpoint, MCPServerProtocol
)

from src.mcp_submission_parsing.servers.agent_parser import ParserAgent
from src.mcp_submission_parsing.servers.agent_policy import PolicyAgent
from src.mcp_submission_parsing.servers.agent_risk import RiskAgent
from src.mcp_submission_parsing.servers.agent_features import FeatureAgent
from src.mcp_submission_parsing.servers.agent_fraud import FraudAgent


class MCPServer(MCPServerProtocol):
    """Main MCP Server that orchestrates all agents"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        logger.info("Initializing MCP Server")
        self.host = host
        self.port = port
        logger.debug(f"Server configuration - Host: {host}, Port: {port}")
        
        self.agents = {}
        self.active_connections = {}
        self.request_history = []
        
        logger.info("Initializing agent handlers")
        self._init_agents()
        logger.info(f"MCP Server initialization complete - {len(self.agents)} agents registered")
    
    def _init_agents(self):
        """Initialize all agent handlers"""
        logger.debug("Starting agent initialization")
        
        self.agents = {
            AgentEndpoint.PARSE_CLAIM: ParserAgent(),
            AgentEndpoint.LOOKUP_POLICY: PolicyAgent(),
            AgentEndpoint.CHECK_RISK: RiskAgent(),
            AgentEndpoint.BUILD_FEATURES: FeatureAgent(),
            AgentEndpoint.DETECT_FRAUD: FraudAgent(),
            AgentEndpoint.COMPLETE_PIPELINE: self._handle_complete_pipeline,
            AgentEndpoint.HEALTH_CHECK: self._handle_health_check,
        }
        
        logger.info(f"Initialized agents: {list(self.agents.keys())}")
        logger.debug("Agent initialization complete")
    
    async def start(self):
        """Start the MCP server"""
        logger.info(f"Starting MCP Server on {self.host}:{self.port}")
        print(f"🚀 Starting MCP Server on {self.host}:{self.port}")
        
        # using stdio for simplicity (can be replaced with HTTP/WebSocket)
        print("📡 Server running in stdio mode")
        print("✅ Ready to accept requests")
        logger.info("Server running in stdio mode")
        
        await self._stdio_loop()
    
    async def _stdio_loop(self):
        """Main loop for stdio communication"""
        logger.info("Entering stdio main loop")
        request_count = 0
        
        while True:
            try:
                # Read message from stdin
                logger.debug("Waiting for incoming request...")
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    logger.warning("Empty line received, breaking loop")
                    break
                
                request_count += 1
                logger.info(f"Received request #{request_count}")
                logger.debug(f"Raw request: {line.strip()}")
                
                # Parse message
                try:
                    message = json.loads(line.strip())
                    logger.debug(f"Parsed JSON message: {message}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}", exc_info=True)
                    error_response = {
                        "error": f"Invalid JSON: {str(e)}",
                        "type": MessageType.ERROR.value
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                    continue
                
                # Process request
                logger.info(f"Processing request with endpoint: {message.get('endpoint')}")
                response = await self.handle_request(message)
                
                # Send response to stdout
                logger.debug(f"Sending response: {json.dumps(response)[:500]}...")  # Truncate for logging
                print(json.dumps(response))
                sys.stdout.flush()
                logger.info(f"Response sent for request #{request_count}")
                
            except Exception as e:
                logger.error(f"Unexpected error in stdio loop: {e}", exc_info=True)
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
        
        logger.info(f"Handling request {request_id} - Endpoint: {endpoint_str}")
        logger.debug(f"Request payload: {payload}")
        
        start_time = time.time()
        
        try:
            endpoint = AgentEndpoint(endpoint_str)
            logger.debug(f"Validated endpoint: {endpoint}")
            
            if endpoint not in self.agents:
                logger.warning(f"Unknown endpoint requested: {endpoint}")
                return {
                    "request_id": request_id,
                    "error": f"Unknown endpoint: {endpoint}",
                    "type": MessageType.ERROR.value
                }
            
            logger.info(f"Routing request {request_id} to agent: {endpoint}")
            # Call agent handler
            result = await self.agents[endpoint](payload)
            logger.debug(f"Agent {endpoint} returned result for request {request_id}")
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Request {request_id} completed in {processing_time:.2f}ms")
            
            # Log request
            history_entry = {
                "request_id": request_id,
                "endpoint": endpoint_str,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            self.request_history.append(history_entry)
            logger.debug(f"Added to request history: {history_entry}")
            
            response = {
                "request_id": request_id,
                "result": result,
                "processing_time_ms": processing_time,
                "type": MessageType.RESPONSE.value
            }
            
            logger.info(f"Successfully processed request {request_id}")
            return response
            
        except ValueError as e:
            logger.error(f"Invalid endpoint value for request {request_id}: {e}")
            return {
                "request_id": request_id,
                "error": f"Invalid endpoint: {endpoint_str}",
                "type": MessageType.ERROR.value
            }
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}", exc_info=True)
            return {
                "request_id": request_id,
                "error": str(e),
                "type": MessageType.ERROR.value
            }
    
    async def _handle_complete_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate complete claim processing pipeline"""
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting complete pipeline processing for request {request_id}")
        logger.debug(f"Pipeline payload: {payload}")
        
        claim_text = payload.get('claim_text', '')
        customer_name = payload.get('customer_name', '')
        customer_id = payload.get('customer_id', '')
        
        logger.info(f"Pipeline inputs - Claim length: {len(claim_text)} chars, Customer: {customer_name} (ID: {customer_id})")
        
        # parse claim
        logger.info(f"[Stage 1/5] Starting claim parsing for request {request_id}")
        stage_start = time.time()
        
        parse_result = await self.agents[AgentEndpoint.PARSE_CLAIM]({
            'text': claim_text,
            'use_llm': True
        })
        
        stage_time = (time.time() - stage_start) * 1000
        logger.info(f"[Stage 1/5] Claim parsing completed in {stage_time:.2f}ms")
        logger.debug(f"Parse result: {parse_result}")
        
        parsed_claim = ParsedClaim(**parse_result) if isinstance(parse_result, dict) else None
        
        if not parsed_claim or not parsed_claim.policy_number:
            logger.error(f"Pipeline {request_id} failed: No policy number extracted")
            return {
                "error": "Failed to extract policy number from claim",
                "status": "REJECTED"
            }
        
        logger.info(f"Extracted policy number: {parsed_claim.policy_number}")
        
        #lookup policy
        logger.info(f"[Stage 2/5] Looking up policy {parsed_claim.policy_number} for request {request_id}")
        stage_start = time.time()
        
        policy_result = await self.agents[AgentEndpoint.LOOKUP_POLICY]({
            'policy_number': parsed_claim.policy_number,
            'incident_date': parsed_claim.incident_date,
            'incident_type': parsed_claim.incident_type,
            'claim_amount': parsed_claim.total_claim_amount
        })
        
        stage_time = (time.time() - stage_start) * 1000
        logger.info(f"[Stage 2/5] Policy lookup completed in {stage_time:.2f}ms")
        logger.debug(f"Policy lookup result: {policy_result}")
        
        policy_verification = PolicyVerification(**policy_result)
        
        if not policy_verification.found:
            logger.warning(f"Pipeline {request_id} failed: Policy {parsed_claim.policy_number} not found")
            return {
                "error": f"Policy {parsed_claim.policy_number} not found",
                "status": "REJECTED"
            }
        
        logger.info(f"Policy verified: Active={policy_verification.is_active}")
        
        # Stage 3: Check risk rules
        logger.info(f"[Stage 3/5] Evaluating risk rules for request {request_id}")
        stage_start = time.time()
        
        risk_result = await self.agents[AgentEndpoint.CHECK_RISK]({
            'parsed': parsed_claim.to_dict(),
            'verification': policy_verification.to_dict()
        })
        
        stage_time = (time.time() - stage_start) * 1000
        logger.info(f"[Stage 3/5] Risk evaluation completed in {stage_time:.2f}ms")
        logger.debug(f"Risk result: {risk_result}")
        
        risk_assessment = RiskAssessment(**risk_result)
        logger.info(f"Risk score: {risk_assessment.risk_score:.3f} ({risk_assessment.risk_level})")
        
        # Stage 4: Build features
        logger.info(f"[Stage 4/5] Building feature vector for request {request_id}")
        stage_start = time.time()
        
        features_result = await self.agents[AgentEndpoint.BUILD_FEATURES]({
            'parsed': parsed_claim.to_dict(),
            'policy': policy_result.get('policy', {}),
            'verification': policy_verification.to_dict(),
            'customer_id': customer_id
        })
        
        stage_time = (time.time() - stage_start) * 1000
        logger.info(f"[Stage 4/5] Feature building completed in {stage_time:.2f}ms")
        logger.debug(f"Features result: {features_result}")
        
        feature_vector = FeatureVector(**features_result)
        logger.info(f"Feature vector contains {len(feature_vector.features)} features")
        
        # Stage 5: Detect fraud
        logger.info(f"[Stage 5/5] Running fraud detection for request {request_id}")
        stage_start = time.time()
        
        fraud_result = await self.agents[AgentEndpoint.DETECT_FRAUD]({
            'features': feature_vector.features,
            'claim_text': claim_text
        })
        
        stage_time = (time.time() - stage_start) * 1000
        logger.info(f"[Stage 5/5] Fraud detection completed in {stage_time:.2f}ms")
        logger.debug(f"Fraud result: {fraud_result}")
        
        fraud_prediction = FraudPrediction(**fraud_result)
        logger.info(f"Fraud probability: {fraud_prediction.fraud_probability:.3f}")
        
        # Generate final decision
        logger.info(f"Generating final decision for request {request_id}")
        final_decision, recommended_action = self._make_final_decision(
            risk_assessment, fraud_prediction
        )
        logger.info(f"Final decision: {final_decision} - {recommended_action}")
        
        # Determine claim status
        if final_decision == "AUTO_APPROVE":
            status = ClaimStatus.APPROVED
            logger.info(f"Claim {request_id} status: APPROVED")
        elif final_decision == "AUTO_DENY":
            status = ClaimStatus.DENIED
            logger.warning(f"Claim {request_id} status: DENIED")
        elif risk_assessment.requires_siu:
            status = ClaimStatus.WITH_SIU
            logger.warning(f"Claim {request_id} status: WITH_SIU (referred to Special Investigations Unit)")
        else:
            status = ClaimStatus.UNDER_REVIEW
            logger.info(f"Claim {request_id} status: UNDER_REVIEW")
        
        # Generate summary
        summary = self._generate_summary(
            parsed_claim, policy_verification, risk_assessment, fraud_prediction
        )
        logger.debug(f"Generated summary: {summary[:200]}...")  # Log first 200 chars
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Pipeline {request_id} completed in {processing_time:.2f}ms total")
        
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
        
        logger.info(f"Returning complete response for request {request_id}")
        return response.to_dict()
    
    async def _handle_health_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Health check endpoint"""
        logger.debug("Health check requested")
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents": list(self.agents.keys()),
            "active_connections": len(self.active_connections),
            "total_requests": len(self.request_history)
        }
        
        logger.info(f"Health check response: {health_status['status']} - {health_status['total_requests']} total requests")
        return health_status
    
    def _make_final_decision(self, risk: RiskAssessment, fraud: FraudPrediction) -> tuple:
        """Make final claim decision based on risk and fraud assessments"""
        
        logger.debug(f"Making final decision - Risk score: {risk.risk_score:.3f}, Fraud prob: {fraud.fraud_probability:.3f}")
        
        if risk.risk_score >= 0.7 or fraud.fraud_probability >= 0.75:
            decision = "AUTO_DENY"
            action = "Refer to SIU with high priority"
            logger.warning(f"High risk/fraud detected - Decision: {decision}")
        elif risk.risk_score >= 0.3 or fraud.fraud_probability >= 0.5:
            decision = "ADJUSTER_REVIEW"
            action = "Send to adjuster for detailed review"
            logger.info(f"Moderate risk/fraud detected - Decision: {decision}")
        elif risk.risk_score <= 0.1 and fraud.fraud_probability <= 0.25:
            decision = "AUTO_APPROVE"
            action = "Auto-approve claim"
            logger.info(f"Low risk/fraud detected - Decision: {decision}")
        else:
            decision = "MANUAL_REVIEW"
            action = "Manual review required"
            logger.info(f"Mixed signals - Decision: {decision}")
        
        logger.debug(f"Final decision: {decision} - {action}")
        return decision, action
    
    def _generate_summary(self, parsed: ParsedClaim, policy: PolicyVerification, 
                          risk: RiskAssessment, fraud: FraudPrediction) -> str:
        """Generate human-readable claim summary"""
        
        logger.debug("Generating human-readable summary")
        
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
        logger.debug(f"Summary generated ({len(summary)} characters)")
        return summary.strip()


async def main():
    """Main entry point"""
    logger.info("Starting MCP Server main entry point")
    print("Initializing MCP Server...")
    
    server = MCPServer()
    
    logger.info("Launching server start routine")
    await server.start()


if __name__ == "__main__":
    logger.info("MCP Server process started")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested (KeyboardInterrupt)")
        print("\n🛑 Server shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("MCP Server process terminated")