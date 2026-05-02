# risk_tools.py
import asyncio
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv() 

class RiskServerClient:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or os.getenv("RISK_SERVER_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("RISK_SERVER_PORT", "8002"))
    
    async def get_tools(self):
        url = f"http://{self.host}:{self.port}/sse"
        
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return tools.tools
    
    async def call_tool(self, tool_name: str, **kwargs):
        url = f"http://{self.host}:{self.port}/sse"
        
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=kwargs)
                return result

# Helper function to run the client
async def test_connection(port: Optional[int] = None):
    client = RiskServerClient(port=port)
    tools = await client.get_tools()
    
    print(f"\n✅ Connected to risk server at http://{client.host}:{client.port}/sse")
    print(f"📦 Found {len(tools)} tools:\n")
    for tool in tools:
        print(f"  🔧 {tool.name}")
        print(f"     {tool.description}")
        print()
    
    return tools

if __name__ == "__main__":
    # Quick test
    asyncio.run(test_connection())