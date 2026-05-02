# test_client.py
import asyncio
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://127.0.0.1:8003/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List all tools
            tools = await session.list_tools()
            print(f"\n✅ Found {len(tools.tools)} tools:\n")
            for tool in tools.tools:
                print(f"  🔧 {tool.name}")
                print(f"     {tool.description}")
                print()

asyncio.run(main())