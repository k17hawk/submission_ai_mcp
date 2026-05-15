import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_ollama_mcp():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "ollama_mcp_server"]
    )
    
    # Correct way: create stdio_client, then ClientSession with read/write streams
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            response = await session.call_tool(
                "llm_extract_fields",
                arguments={"text": "My policy POL-123456, accident on 2025-01-15, claim $4500"}
            )
            print("Response:", response)

if __name__ == "__main__":
    asyncio.run(test_ollama_mcp())