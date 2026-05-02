"""
Risk Checking MCP Server
Provides clause rating and risk assessment tools for insurance submissions.
"""

import sys
from pathlib import Path


from mcp.server.fastmcp import FastMCP
from src.mcp_risk_checking.tools.pipeline import (
    analyze_clause_risk,
    analyze_submission_text,
)
from src.mcp_risk_checking.tools.rating import get_available_categories
mcp = FastMCP("Insurance Submission Parser")
mcp.add_tool(analyze_clause_risk)
mcp.add_tool(analyze_submission_text)


# ─── Resource: Available Categories ───
@mcp.resource("risk://categories")
def list_categories() -> str:
    import json
    import asyncio
    categories = asyncio.run(get_available_categories())
    return json.dumps({"total": len(categories), "categories": categories}, indent=2)

# ─── Run ───
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Risk Checking MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.transport == "sse":
        import uvicorn
        try:
            app = mcp.sse_app()
        except AttributeError:
            app = mcp._asgi_app
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        mcp.run()

