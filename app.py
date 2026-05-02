# api_server.py
"""
FastAPI server for the Underwriting Assistant
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mcp_client import mcp_manager
from langgraph_underwriting import build_underwriting_graph, run_underwriting_pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Underwriting Assistant",
    description="API for processing insurance submissions using MCP servers and LangGraph",
    version="1.0.0"
)

# Store for background tasks and results
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

processing_jobs = {}


# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════

class SubmissionResponse(BaseModel):
    submission_id: str
    status: str
    message: str

class ProcessingStatus(BaseModel):
    submission_id: str
    status: str  # "processing", "completed", "failed"
    progress: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None

class UnderwritingReport(BaseModel):
    submission_id: str
    policy_number: Optional[str]
    insured_name: Optional[str]
    policy_type: Optional[str]
    overall_risk: Optional[str]
    average_rating: Optional[float]
    final_decision: Optional[str]
    decision_emoji: Optional[str]
    executive_summary: str
    full_report: str
    agent_logs: List[str]
    errors: List[str]
    generated_at: str
    processing_time_seconds: float


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """Home page with API documentation"""
    return """
    <html>
    <head>
        <title>Underwriting Assistant API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2563eb; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #2563eb; }
            .method { font-weight: bold; color: #059669; }
            code { background: #e5e7eb; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Multi-Agent Underwriting Assistant</h1>
            <p>Process insurance submissions using AI agents</p>
            
            <h2>Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/api/submit</code>
                <p>Upload and process an insurance submission PDF</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/status/{submission_id}</code>
                <p>Check processing status</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/report/{submission_id}</code>
                <p>Get the final underwriting report</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/health</code>
                <p>Check server and MCP connections health</p>
            </div>
            
            <h2>Quick Test</h2>
            <form action="/api/submit" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf">
                <button type="submit">Process PDF</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.get("/api/health")
async def health_check():
    """Check if API and MCP servers are running"""
    mcp_status = await mcp_manager.verify_servers()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mcp_servers": mcp_status
    }


@app.post("/api/submit", response_model=SubmissionResponse)
async def submit_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload an ACORD PDF submission for processing.
    Returns immediately with a submission_id for tracking.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are accepted")
    
    # Generate unique submission ID
    submission_id = str(uuid.uuid4())[:8]
    
    # Save file
    pdf_path = UPLOAD_DIR / f"{submission_id}_{file.filename}"
    content = await file.read()
    pdf_path.write_bytes(content)
    
    # Initialize job tracking
    processing_jobs[submission_id] = {
        "status": "queued",
        "progress": "Queued for processing",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
        "pdf_path": str(pdf_path),
        "filename": file.filename
    }
    
    # Start processing in background
    background_tasks.add_task(process_submission, submission_id, str(pdf_path))
    
    return SubmissionResponse(
        submission_id=submission_id,
        status="queued",
        message=f"PDF received. Track progress at /api/status/{submission_id}"
    )


@app.get("/api/status/{submission_id}", response_model=ProcessingStatus)
async def get_status(submission_id: str):
    """Check processing status of a submission"""
    if submission_id not in processing_jobs:
        raise HTTPException(404, "Submission not found")
    
    job = processing_jobs[submission_id]
    return ProcessingStatus(
        submission_id=submission_id,
        status=job["status"],
        progress=job["progress"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error")
    )


@app.get("/api/report/{submission_id}", response_model=UnderwritingReport)
async def get_report(submission_id: str):
    """Get the final underwriting report"""
    if submission_id not in processing_jobs:
        raise HTTPException(404, "Submission not found")
    
    job = processing_jobs[submission_id]
    
    if job["status"] != "completed":
        raise HTTPException(400, f"Processing not complete. Status: {job['status']}")
    
    result = job["result"]
    
    # Calculate processing time
    start = datetime.fromisoformat(job["started_at"])
    end = datetime.fromisoformat(job["completed_at"])
    processing_time = (end - start).total_seconds()
    
    return UnderwritingReport(
        submission_id=submission_id,
        policy_number=result.get("policy_data", {}).get("policy_number"),
        insured_name=result.get("policy_data", {}).get("insured_name"),
        policy_type=result.get("policy_data", {}).get("policy_type"),
        overall_risk=result.get("risk_assessment", {}).get("overall_risk") if isinstance(result.get("risk_assessment"), dict) else None,
        average_rating=result.get("risk_assessment", {}).get("average_rating") if isinstance(result.get("risk_assessment"), dict) else None,
        final_decision=result.get("final_decision"),
        decision_emoji=result.get("decision_emoji"),
        executive_summary=result.get("executive_summary", ""),
        full_report=result.get("full_report", ""),
        agent_logs=result.get("agent_logs", []),
        errors=result.get("errors", []),
        generated_at=result.get("report_generated_at", ""),
        processing_time_seconds=processing_time
    )


@app.delete("/api/submission/{submission_id}")
async def delete_submission(submission_id: str):
    """Delete a submission and its files"""
    if submission_id in processing_jobs:
        job = processing_jobs[submission_id]
        pdf_path = Path(job["pdf_path"])
        if pdf_path.exists():
            pdf_path.unlink()
        del processing_jobs[submission_id]
        return {"message": "Submission deleted"}
    raise HTTPException(404, "Submission not found")


# ═══════════════════════════════════════════════════════════════
# BACKGROUND PROCESSING
# ═══════════════════════════════════════════════════════════════

async def process_submission(submission_id: str, pdf_path: str):
    """Background task to process a submission"""
    job = processing_jobs[submission_id]
    
    try:
        job["status"] = "processing"
        job["progress"] = "Starting multi-agent pipeline..."
        
        # Run the LangGraph pipeline
        result = await run_underwriting_pipeline(
            pdf_path=pdf_path,
            submission_id=submission_id
        )
        
        # Update job with results
        job["status"] = "completed"
        job["progress"] = "Processing complete"
        job["completed_at"] = datetime.now().isoformat()
        job["result"] = result
        
        logger.info(f"✅ Submission {submission_id} processed successfully")
        
    except Exception as e:
        job["status"] = "failed"
        job["progress"] = f"Failed: {str(e)}"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        logger.error(f"❌ Submission {submission_id} failed: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════════
# STARTUP/SHUTDOWN
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def startup():
    """Verify MCP servers on startup"""
    logger.info("🚀 Starting Underwriting Assistant API...")
    status = await mcp_manager.verify_servers()
    
    for server, info in status.items():
        logger.info(f"  {server}: {info['status']}")
    
    if all("✅" in info["status"] for info in status.values()):
        logger.info("✅ All MCP servers connected")
    else:
        logger.warning("⚠️ Some MCP servers may not be available")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "9000"))
    host = os.getenv("API_HOST", "127.0.0.1")
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")