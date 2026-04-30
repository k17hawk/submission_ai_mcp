# server.py
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP

# Existing tool imports
from src.mcp_insurance.tools.parsing import parse_acord_submission, extract_policy_data
from src.mcp_insurance.tools.retrieval import search_corpus, get_document_by_id
from src.mcp_insurance.tools.evaluation import evaluate_retrieval
from src.mcp_insurance.tools.rating import rate_clause, get_rating_examples

# New: pipeline tool and helper for rating categories
from src.mcp_insurance.tools.pipeline import process_submission
from src.mcp_insurance.tools.rating import get_available_categories

# Create MCP server
mcp = FastMCP("Insurance Submission Parser")

# ----------- register existing tools -----------
mcp.add_tool(parse_acord_submission)
mcp.add_tool(extract_policy_data)
mcp.add_tool(search_corpus)
mcp.add_tool(get_document_by_id)
mcp.add_tool(evaluate_retrieval)
mcp.add_tool(rate_clause)
mcp.add_tool(get_rating_examples)

# ----------- register new pipeline tool -----------
mcp.add_tool(process_submission)


@mcp.resource("rating://categories")
def list_rating_categories() -> str:
    import json
    categories = get_available_categories()
    logger.info(f"✅ Resource 'rating://categories' was read – returning {len(categories)} categories")
    return json.dumps(categories, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
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