# api_server.py
"""
FastAPI server for the Underwriting Assistant
"""

import asyncio
import logging
import os
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for background tasks and results
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

processing_jobs = {}


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def safe_extract(obj: Any, default: Any = None) -> Any:
    """
    Safely extract value from objects that might be CallToolResult or other MCP types.
    """
    if obj is None:
        return default
    
    # If it's already a basic type, return as is
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # If it's a dict, return as is
    if isinstance(obj, dict):
        return obj
    
    # If it's a list, return as is
    if isinstance(obj, list):
        return obj
    
    # Handle CallToolResult or similar objects
    if hasattr(obj, 'content'):
        content = obj.content
        if isinstance(content, list) and len(content) > 0:
            if hasattr(content[0], 'text'):
                try:
                    return json.loads(content[0].text)
                except (json.JSONDecodeError, AttributeError):
                    return content[0].text
            return content[0]
        return content
    
    # Try to convert to dict if it has dict() method
    try:
        return obj.dict()
    except (AttributeError, TypeError):
        pass
    
    # Try to convert to string
    try:
        return str(obj)
    except:
        return default


def safe_dict_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a key from an object that might be CallToolResult or dict.
    """
    # First extract the actual data
    data = safe_extract(obj)
    
    # Now try to get the key
    if isinstance(data, dict):
        return data.get(key, default)
    
    return default


def normalize_result(result: Any) -> Dict[str, Any]:
    """
    Normalize the pipeline result to ensure it's a proper dictionary.
    """
    # Extract the actual data
    data = safe_extract(result)
    
    if not isinstance(data, dict):
        logger.warning(f"Result is not a dict after extraction: {type(data)}")
        return {
            "executive_summary": str(data),
            "full_report": str(data),
            "agent_logs": [],
            "errors": [f"Unexpected result type: {type(data)}"],
            "report_generated_at": datetime.now().isoformat(),
            "final_decision": "Unknown",
            "decision_emoji": "❓"
        }
    
    return data


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
    policy_number: Optional[str] = None
    insured_name: Optional[str] = None
    policy_type: Optional[str] = None
    overall_risk: Optional[str] = None
    average_rating: Optional[float] = None
    final_decision: Optional[str] = None
    decision_emoji: Optional[str] = None
    executive_summary: str
    full_report: str
    agent_logs: List[str] = []
    errors: List[str] = []
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
    
    # Ensure all values are JSON serializable
    normalized_status = {}
    for server, info in mcp_status.items():
        normalized_status[server] = safe_extract(info, {"status": "unknown"})
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mcp_servers": normalized_status
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


@app.get("/api/report/{submission_id}")
async def get_report(submission_id: str):
    """Get the final underwriting report"""
    if submission_id not in processing_jobs:
        raise HTTPException(404, "Submission not found")
    
    job = processing_jobs[submission_id]
    
    if job["status"] != "completed":
        raise HTTPException(400, f"Processing not complete. Status: {job['status']}")
    
    result = job["result"]
    
    # Normalize the result to ensure it's a proper dictionary
    result = normalize_result(result)
    
    # Calculate processing time
    start = datetime.fromisoformat(job["started_at"])
    end = datetime.fromisoformat(job["completed_at"])
    processing_time = (end - start).total_seconds()
    
    # Safely extract nested values
    policy_data = safe_extract(result.get("policy_data"), {})
    risk_assessment = safe_extract(result.get("risk_assessment"), {})
    
    # Build the report
    report = {
        "submission_id": submission_id,
        "policy_number": safe_dict_get(policy_data, "policy_number"),
        "insured_name": safe_dict_get(policy_data, "insured_name"),
        "policy_type": safe_dict_get(policy_data, "policy_type"),
        "overall_risk": safe_dict_get(risk_assessment, "overall_risk") if isinstance(risk_assessment, dict) else None,
        "average_rating": safe_dict_get(risk_assessment, "average_rating") if isinstance(risk_assessment, dict) else None,
        "final_decision": safe_dict_get(result, "final_decision", "Unknown"),
        "decision_emoji": safe_dict_get(result, "decision_emoji", "❓"),
        "executive_summary": safe_dict_get(result, "executive_summary", ""),
        "full_report": safe_dict_get(result, "full_report", ""),
        "agent_logs": safe_extract(result.get("agent_logs"), []),
        "errors": safe_extract(result.get("errors"), []),
        "generated_at": safe_dict_get(result, "report_generated_at", datetime.now().isoformat()),
        "processing_time_seconds": processing_time
    }
    
    # Ensure agent_logs and errors are lists
    if not isinstance(report["agent_logs"], list):
        report["agent_logs"] = [str(report["agent_logs"])]
    if not isinstance(report["errors"], list):
        report["errors"] = [str(report["errors"])]
    
    # Ensure strings for text fields
    for field in ["executive_summary", "full_report", "final_decision", "decision_emoji"]:
        if not isinstance(report[field], str):
            report[field] = str(report[field]) if report[field] is not None else ""
    
    # Ensure numeric fields
    if report["average_rating"] is not None:
        try:
            report["average_rating"] = float(report["average_rating"])
        except (ValueError, TypeError):
            report["average_rating"] = None
    
    return report


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
        
        logger.info(f"Processing submission {submission_id}...")
        
        # Run the LangGraph pipeline
        result = await run_underwriting_pipeline(
            pdf_path=pdf_path,
            submission_id=submission_id
        )
        
        # Normalize the result
        normalized_result = normalize_result(result)
        
        # Update job with results
        job["status"] = "completed"
        job["progress"] = "Processing complete"
        job["completed_at"] = datetime.now().isoformat()
        job["result"] = normalized_result
        
        logger.info(f"✅ Submission {submission_id} processed successfully")
        logger.info(f"Result keys: {list(normalized_result.keys()) if isinstance(normalized_result, dict) else 'Not a dict'}")
        
    except Exception as e:
        job["status"] = "failed"
        job["progress"] = f"Failed: {str(e)}"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        logger.error(f"❌ Submission {submission_id} failed: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════════
# STARTUP/SHUTDOWN
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Verify MCP servers on startup"""
    logger.info("🚀 Starting Underwriting Assistant API...")
    
    # Add CORS headers to all responses
    logger.info("CORS middleware enabled")
    
    try:
        status = await mcp_manager.verify_servers()
        for server, info in status.items():
            server_status = safe_extract(info, {})
            status_text = server_status.get("status", "unknown") if isinstance(server_status, dict) else str(server_status)
            logger.info(f"  {server}: {status_text}")
        
        if all("✅" in str(safe_dict_get(info, "status", "")) for info in status.values()):
            logger.info("✅ All MCP servers connected")
        else:
            logger.warning("⚠️ Some MCP servers may not be available")
    except Exception as e:
        logger.warning(f"Could not verify MCP servers: {e}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "9000"))
    host = os.getenv("API_HOST", "127.0.0.1")
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")