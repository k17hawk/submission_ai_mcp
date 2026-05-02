# insurance_client.py
import asyncio
import os
from typing import Optional
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

class InsuranceServerClient:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or os.getenv("SUBMISSION_PARSER_HOST", os.getenv("MCP_HOST", "127.0.0.1"))
        self.port = port or int(os.getenv("SUBMISSION_PARSER_PORT", os.getenv("MCP_PORT", "8008")))
    
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
    
    async def get_resources(self):
        """List available resources from the server"""
        url = f"http://{self.host}:{self.port}/sse"
        
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resources = await session.list_resources()
                return resources
    
    async def read_resource(self, uri: str):
        """Read a specific resource from the server"""
        url = f"http://{self.host}:{self.port}/sse"
        
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.read_resource(uri)
                return result

# Helper function to test connection
async def test_connection(port: Optional[int] = None):
    client = InsuranceServerClient(port=port)
    
    print(f"🔍 Connecting to insurance server at http://{client.host}:{client.port}/sse")
    
    tools = await client.get_tools()
    print(f"\n✅ Connected successfully!")
    print(f"📦 Found {len(tools)} tools:\n")
    for tool in tools:
        print(f"  🔧 {tool.name}")
        if tool.description:
            print(f"     {tool.description}")
        print()
    
    # Also check resources
    try:
        resources = await client.get_resources()
        if resources:
            print(f"📚 Found {len(resources)} resources:")
            for resource in resources:
                print(f"  📄 {resource.uri}")
                if resource.description:
                    print(f"     {resource.description}")
    except Exception:
        pass  # Resources might not be supported
    
    return client

if __name__ == "__main__":
    asyncio.run(test_connection())