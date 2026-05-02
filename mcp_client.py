# mcp_clients.py
"""
Unified MCP Client Manager for both Risk and Insurance servers.
Uses proper MCP client libraries instead of direct imports.
"""

import asyncio
import os
import logging
from typing import Optional, Dict, Any, List
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MCPServerClient:
    """Async client for an MCP server"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, name: str = "mcp"):
        self.host = host
        self.port = port
        self.name = name
        self.url = f"http://{host}:{port}/sse"
    
    @asynccontextmanager
    async def session(self):
        """Get an initialized MCP session"""
        async with sse_client(self.url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool on this MCP server"""
        async with self.session() as session:
            result = await session.call_tool(tool_name, arguments=kwargs)
            return result
    
    async def list_tools(self) -> List[Dict]:
        """List available tools"""
        async with self.session() as session:
            tools = await session.list_tools()
            return [{"name": t.name, "description": t.description} for t in tools.tools]


class MCPManager:
    """Manages connections to both MCP servers"""
    
    def __init__(self):
        # Server 1: Insurance Submission Parser
        self.insurance = MCPServerClient(
            host=os.getenv("INSURANCE_SERVER_HOST", "127.0.0.1"),
            port=int(os.getenv("INSURANCE_SERVER_PORT", "8008")),
            name="insurance"
        )
        
        # Server 2: Risk Checker
        self.risk = MCPServerClient(
            host=os.getenv("RISK_SERVER_HOST", "127.0.0.1"),
            port=int(os.getenv("RISK_SERVER_PORT", "8004")),
            name="risk"
        )
    
    # ─── Insurance Server Methods ───
    
    async def parse_submission(self, pdf_path: str) -> Dict[str, Any]:
        """Parse ACORD PDF submission"""
        logger.info(f"📄 Parsing submission: {pdf_path}")
        return await self.insurance.call_tool("parse_acord_submission", pdf_path=pdf_path)
    
    async def search_corpus(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search insurance clause corpus"""
        return await self.insurance.call_tool(
            "search_corpus", 
            query=query, 
            top_k=top_k
        )
    
    async def rate_clause(self, clause_text: str, category: str = None) -> Dict:
        """Rate a clause using insurance server"""
        kwargs = {"clause_text": clause_text}
        if category:
            kwargs["category"] = category
        return await self.insurance.call_tool("rate_clause", **kwargs)
    
    async def detect_categories(self, text: str) -> List[str]:
        """Detect clause categories from text"""
        # This uses the internal function from the server
        # We'll call process_submission with auto-detect
        result = await self.insurance.call_tool(
            "process_submission",
            pdf_path="",  # Not used when we have text
            full_text=text
        )
        # Extract categories from result
        categories = []
        if result and isinstance(result, dict):
            for item in result.get("results", []):
                if "query" in item:
                    categories.append(item["query"])
        return categories
    
    # ─── Risk Server Methods ───
    
    async def analyze_submission_risk(self, 
                                      full_text: str, 
                                      policy_type: str = None) -> Dict:
        """Analyze submission risk using risk server"""
        logger.info("🔍 Analyzing submission risk (Risk Server)")
        kwargs = {"full_text": full_text}
        if policy_type:
            kwargs["policy_type"] = policy_type
        return await self.risk.call_tool("analyze_submission_text", **kwargs)
    
    async def analyze_clause_risk(self,
                                  clause_text: str,
                                  policy_type: str = None) -> Dict:
        """Analyze single clause risk"""
        kwargs = {"clause_text": clause_text}
        if policy_type:
            kwargs["policy_type"] = policy_type
        return await self.risk.call_tool("analyze_clause_risk", **kwargs)
    
    async def verify_servers(self) -> Dict[str, Any]:
        """Verify both servers are accessible"""
        result = {}
        
        # Check insurance server
        try:
            tools = await self.insurance.list_tools()
            result["insurance"] = {
                "status": "✅ Connected",
                "url": self.insurance.url,
                "tools": len(tools),
                "tool_names": [t["name"] for t in tools]
            }
        except Exception as e:
            result["insurance"] = {
                "status": f"❌ Failed: {e}",
                "url": self.insurance.url
            }
        
        # Check risk server
        try:
            tools = await self.risk.list_tools()
            result["risk"] = {
                "status": "✅ Connected",
                "url": self.risk.url,
                "tools": len(tools),
                "tool_names": [t["name"] for t in tools]
            }
        except Exception as e:
            result["risk"] = {
                "status": f"❌ Failed: {e}",
                "url": self.risk.url
            }
        
        return result

mcp_manager = MCPManager()