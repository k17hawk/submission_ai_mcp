# src/fastapi_server.py
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import your LangGraph workflow
from src.lang_graph.workflow import graph
from src.lang_graph.state import ClaimProcessingState

# ----------------------------------------------------------------------
# In‑memory store for active/past requests
# In production, replace with Redis, PostgreSQL, etc.
# ----------------------------------------------------------------------
store: Dict[str, Dict[str, Any]] = {}

# ----------------------------------------------------------------------
# Pydantic models for requests/responses
# ----------------------------------------------------------------------
class ClaimProcessRequest(BaseModel):
    claim_text: str = Field(..., description="Raw claim description")
    customer_name: str = Field(..., description="Full name of the customer")
    customer_id: str = Field(..., description="Customer identifier (e.g., CUST-XXXXX)")

class ClaimProcessResponse(BaseModel):
    request_id: str
    status: str  # "processing", "completed", "failed"
    message: str = "Claim processing started"

class ClaimStatusResponse(BaseModel):
    request_id: str
    status: str
    final_decision: Optional[str] = None
    recommended_action: Optional[str] = None
    errors: list = []
    processing_time_ms: float = 0.0
    updated_at: str

class ClaimResultResponse(BaseModel):
    request_id: str
    final_state: Dict[str, Any]

# ----------------------------------------------------------------------
# Lifespan manager (optional: load anything at startup)
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: any initialisation (e.g., warm up agents)
    print("🚀 FastAPI server starting...")
    yield
    # Shutdown: cleanup
    print("🛑 FastAPI server shutting down...")

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(
    title="Insurance Claim Processing API",
    description="LangGraph‑powered claim workflow with parsing, policy lookup, risk, features, fraud",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for your Streamlit client (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development – restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Background task runner
# ----------------------------------------------------------------------
async def run_claim_workflow(request_id: str, initial_state: ClaimProcessingState):
    """Execute the LangGraph workflow and store the final state."""
    try:
        config = {"configurable": {"thread_id": request_id}}
        final_state = await graph.ainvoke(initial_state, config=config)
        
        # Update store with final result
        store[request_id]["status"] = "completed"
        store[request_id]["final_state"] = final_state
        store[request_id]["final_decision"] = final_state.get("final_decision")
        store[request_id]["recommended_action"] = final_state.get("recommended_action")
        store[request_id]["errors"] = final_state.get("errors", [])
        store[request_id]["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        store[request_id]["status"] = "failed"
        store[request_id]["errors"] = [str(e)]
        store[request_id]["updated_at"] = datetime.now().isoformat()

# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------
@app.post("/claims/process", response_model=ClaimProcessResponse)
async def process_claim(request: ClaimProcessRequest, background_tasks: BackgroundTasks):
    """
    Start processing a new insurance claim.
    Returns a request_id that can be used to poll status and result.
    """
    request_id = str(uuid.uuid4())
    
    # Build initial state matching the LangGraph state schema
    initial_state: ClaimProcessingState = {
        "claim_text": request.claim_text,
        "customer_name": request.customer_name,
        "customer_id": request.customer_id,
        "parsed_claim": None,
        "policy_verification": None,
        "risk_assessment": None,
        "feature_vector": None,
        "fraud_prediction": None,
        "final_decision": None,
        "recommended_action": None,
        "status": None,
        "missing_fields": [],
        "errors": [],
        "processing_time_ms": 0.0,
        "llm_enhanced": False,
        "extraction_success": False,
        "policy_found": False,
        "critical_missing": []
    }
    
    # Store initial metadata
    store[request_id] = {
        "status": "processing",
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "final_state": None,
        "final_decision": None,
        "recommended_action": None,
        "errors": []
    }
    
    # Run workflow in background
    background_tasks.add_task(run_claim_workflow, request_id, initial_state)
    
    return ClaimProcessResponse(
        request_id=request_id,
        status="processing",
        message="Claim processing started. Use /claims/{request_id}/status to track progress."
    )

@app.get("/claims/{request_id}/status", response_model=ClaimStatusResponse)
async def get_claim_status(request_id: str):
    """Get current status of a claim processing request."""
    if request_id not in store:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    data = store[request_id]
    return ClaimStatusResponse(
        request_id=request_id,
        status=data["status"],
        final_decision=data.get("final_decision"),
        recommended_action=data.get("recommended_action"),
        errors=data.get("errors", []),
        processing_time_ms=data.get("processing_time_ms", 0.0),
        updated_at=data["updated_at"]
    )

@app.get("/claims/{request_id}/result", response_model=ClaimResultResponse)
async def get_claim_result(request_id: str):
    """Retrieve the complete final state of a completed claim."""
    if request_id not in store:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    data = store[request_id]
    if data["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Claim not finished yet. Current status: {data['status']}")
    
    return ClaimResultResponse(
        request_id=request_id,
        final_state=data.get("final_state", {})
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)