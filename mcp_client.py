# mcp_clients.py
"""
Unified MCP Client Manager for both Risk and Insurance servers.
Handles proper extraction of content from MCP CallToolResult objects.
"""

import asyncio
import os
import json
import logging
from typing import Optional, Dict, Any, List, Union
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


def extract_result(result: Any) -> Any:
    """
    Extract actual content from MCP CallToolResult.
    MCP returns CallToolResult objects which contain content items.
    """
   
    if isinstance(result, dict):
        return result
    
    # If it's a list, return as-is
    if isinstance(result, list):
        return result
    
    # If it's a string, try to parse as JSON
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    
    # Handle CallToolResult object
    if hasattr(result, 'content'):
        content = result.content
        
        # If content is a list of TextContent objects
        if isinstance(content, list):
            texts = []
            for item in content:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    texts.append(json.dumps(item))
            
            combined_text = "\n".join(texts)
            
            # Try to parse as JSON
            try:
                return json.loads(combined_text)
            except (json.JSONDecodeError, TypeError):
                return combined_text
        
        # If content is a string
        if isinstance(content, str):
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return content
        
        return content
    
    # Handle result with .result attribute
    if hasattr(result, 'result'):
        return extract_result(result.result)
    
    # Fallback: convert to string
    return str(result)


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
        """
        Call a tool on this MCP server and extract the actual result.
        Handles CallToolResult properly.
        """
        logger.info(f"🔧 Calling {self.name}/{tool_name} with args: {list(kwargs.keys())}")
        
        try:
            async with self.session() as session:
                result = await session.call_tool(tool_name, arguments=kwargs)
                
                # Extract the actual content from the result
                extracted = extract_result(result)
                
                logger.info(f"✅ {self.name}/{tool_name} completed successfully")
                logger.debug(f"   Result type: {type(extracted)}")
                if isinstance(extracted, dict):
                    logger.debug(f"   Keys: {list(extracted.keys())[:5]}")
                
                return extracted
                
        except Exception as e:
            logger.error(f"❌ {self.name}/{tool_name} failed: {e}")
            raise
    
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
        result = await self.insurance.call_tool("parse_acord_submission", pdf_path=pdf_path)
        
        # Ensure result is a dict
        if isinstance(result, str):
            return {"text": result, "policy_data": {}}
        return result if isinstance(result, dict) else {"text": str(result), "policy_data": {}}
    
    async def extract_policy_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract only policy data"""
        result = await self.insurance.call_tool("extract_policy_data", pdf_path=pdf_path)
        return result if isinstance(result, dict) else {}
    
    async def search_corpus(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search insurance clause corpus"""
        result = await self.insurance.call_tool(
            "search_corpus", 
            query=query, 
            top_k=top_k
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("results", [])
        return []
    
    async def rate_clause(self, clause_text: str, category: str = None) -> Dict:
        """Rate a clause using insurance server"""
        kwargs = {"clause_text": clause_text}
        if category:
            kwargs["category"] = category
        result = await self.insurance.call_tool("rate_clause", **kwargs)
        return result if isinstance(result, dict) else {"result": str(result)}
    
    async def process_submission(self, pdf_path: str, **kwargs) -> Dict:
        """Full submission processing pipeline"""
        result = await self.insurance.call_tool("process_submission", pdf_path=pdf_path, **kwargs)
        return result if isinstance(result, dict) else {"result": str(result)}
    
    async def get_rating_examples(self, category: str, top_k: int = 3) -> Dict:
        """Get example ratings for a category"""
        result = await self.insurance.call_tool(
            "get_rating_examples",
            category=category,
            top_k=top_k
        )
        return result if isinstance(result, dict) else {"examples": []}
    
    # ─── Risk Server Methods ───
    
    async def analyze_submission_risk(self, 
                                      full_text: str, 
                                      policy_type: str = None) -> Dict:
        """
        Analyze submission risk using Risk Server.
        Uses analyze_submission_text for full submission analysis.
        """
        logger.info("🔍 Analyzing submission risk (Risk Server)")
        kwargs = {"full_text": full_text}
        if policy_type:
            kwargs["policy_type"] = policy_type
        
        result = await self.risk.call_tool("analyze_submission_text", **kwargs)
        
        # Ensure result is a dict with expected structure
        if isinstance(result, str):
            return {
                "overall_risk": "UNKNOWN",
                "average_rating": 0,
                "clauses": [],
                "raw_result": result
            }
        return result if isinstance(result, dict) else {"raw_result": str(result)}
    
    async def analyze_clause_risk(self,
                                  clause_text: str,
                                  policy_type: str = None,
                                  auto_detect_categories: bool = True) -> Dict:
        """Analyze single clause risk using Risk Server"""
        kwargs = {
            "clause_text": clause_text,
            "auto_detect_categories": auto_detect_categories
        }
        if policy_type:
            kwargs["policy_type"] = policy_type
        
        result = await self.risk.call_tool("analyze_clause_risk", **kwargs)
        return result if isinstance(result, dict) else {"raw_result": str(result)}
    
    # ─── Utility Methods ───
    
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
            logger.info(f"✅ Insurance server: {len(tools)} tools available")
        except Exception as e:
            result["insurance"] = {
                "status": f"❌ Failed: {str(e)}",
                "url": self.insurance.url,
                "tools": 0,
                "tool_names": []
            }
            logger.warning(f"⚠️ Insurance server unavailable: {e}")
        
        # Check risk server
        try:
            tools = await self.risk.list_tools()
            result["risk"] = {
                "status": "✅ Connected",
                "url": self.risk.url,
                "tools": len(tools),
                "tool_names": [t["name"] for t in tools]
            }
            logger.info(f"✅ Risk server: {len(tools)} tools available")
        except Exception as e:
            result["risk"] = {
                "status": f"❌ Failed: {str(e)}",
                "url": self.risk.url,
                "tools": 0,
                "tool_names": []
            }
            logger.warning(f"⚠️ Risk server unavailable: {e}")
        
        return result
    
    async def get_resources(self, server: str = "risk") -> Any:
        """Get resources from a server"""
        client = self.risk if server == "risk" else self.insurance
        
        async with client.session() as session:
            resources = await session.list_resources()
            return resources


# ═══════════════════════════════════════════════════════════════
# Singleton instance
# ═══════════════════════════════════════════════════════════════

mcp_manager = MCPManager()


# ═══════════════════════════════════════════════════════════════
# Test function
# ═══════════════════════════════════════════════════════════════

async def test_connections():
    """Test all MCP connections and tools"""
    print("=" * 60)
    print("🧪 Testing MCP Server Connections")
    print("=" * 60)
    
    # Test server connectivity
    status = await mcp_manager.verify_servers()
    
    for server, info in status.items():
        print(f"\n📡 {server.upper()} SERVER: {info['url']}")
        print(f"   Status: {info['status']}")
        if info['tool_names']:
            print(f"   Tools ({info['tools']}):")
            for tool_name in info['tool_names']:
                print(f"     🔧 {tool_name}")
    
    # Test a simple tool call if servers are up
    all_ok = all("✅" in info["status"] for info in status.values())
    
    if all_ok:
        print("\n" + "=" * 60)
        print("🔄 Testing Tool Calls...")
        print("=" * 60)
        
        # Test insurance server
        try:
            print("\n📄 Testing search_corpus on insurance server...")
            result = await mcp_manager.search_corpus("cap on liability", top_k=1)
            print(f"   ✅ Got {len(result) if isinstance(result, list) else '?'} results")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test risk server
        try:
            print("\n⚠️ Testing analyze_clause_risk on risk server...")
            result = await mcp_manager.analyze_clause_risk(
                "The liability cap shall not exceed $1,000,000.",
                policy_type="commercial_general_liability"
            )
            if isinstance(result, dict):
                print(f"   ✅ Result keys: {list(result.keys())[:5]}")
            else:
                print(f"   ✅ Got result: {str(result)[:100]}...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Test complete")
    return status


if __name__ == "__main__":
    asyncio.run(test_connections())