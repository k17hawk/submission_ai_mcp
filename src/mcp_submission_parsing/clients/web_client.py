#!/usr/bin/env python3
"""
Web Client for MCP Insurance System using FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_submission_parsing.clients.cli_client import MCPInsuranceClient


# Pydantic models for request/response
class ParseRequest(BaseModel):
    text: str
    use_llm: bool = True
    doc_type: str = "unknown"


class ProcessRequest(BaseModel):
    claim_text: str
    customer_name: Optional[str] = None
    customer_id: Optional[str] = None


class ClaimResponse(BaseModel):
    request_id: str
    status: str
    final_decision: str
    risk_score: Optional[float] = None
    fraud_probability: Optional[float] = None
    summary: Optional[str] = None
    error: Optional[str] = None


# Create FastAPI app
app = FastAPI(
    title="MCP Insurance Claims API",
    description="API for processing insurance claims through MCP agents",
    version="1.0.0"
)

# Initialize client
client = MCPInsuranceClient()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MCP Insurance Claims Processing",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/parse",
            "/process",
            "/process/complete"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        result = await client.health_check()
        return {"status": "healthy", "details": result}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/parse")
async def parse_claim(request: ParseRequest):
    """Parse a claim submission"""
    try:
        result = await client.parse_claim(request.text, request.use_llm)
        return result.get('result', result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=ClaimResponse)
async def process_claim(request: ProcessRequest):
    """Process a complete claim"""
    try:
        result = await client.process_complete_claim(
            request.claim_text,
            request.customer_name or ""
        )
        
        if 'error' in result:
            return ClaimResponse(
                request_id=result.get('request_id', 'unknown'),
                status='ERROR',
                final_decision='REJECTED',
                error=result['error']
            )
        
        return ClaimResponse(
            request_id=result.get('request_id', 'unknown'),
            status=result.get('status', 'UNKNOWN'),
            final_decision=result.get('final_decision', 'UNKNOWN'),
            risk_score=result.get('risk_assessment', {}).get('risk_score'),
            fraud_probability=result.get('fraud_prediction', {}).get('fraud_probability'),
            summary=result.get('summary')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/batch")
async def process_batch(claims: list[ProcessRequest]):
    """Process multiple claims in batch"""
    results = []
    for claim in claims:
        result = await process_claim(claim)
        results.append(result.dict())
    return {"total": len(results), "results": results}


def main():
    """Run the web server"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()