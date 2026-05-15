#!/usr/bin/env python3
"""
Command Line Client for MCP Insurance System
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.protocols import AgentEndpoint
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPInsuranceClient:
    """Client for interacting with MCP Insurance Server"""
    
    def __init__(self):
        self.server_params = StdioServerParameters(
            command="python",
            args=["servers/main_server.py"]
        )
    
    async def call_tool(self, endpoint: AgentEndpoint, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool/endpoint"""
        
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # Prepare request
                request = {
                    "request_id": "cli-001",
                    "endpoint": endpoint.value,
                    "payload": payload
                }
                
                # Send request
                response = await session.call_tool(
                    "process_request",
                    arguments={"request": json.dumps(request)}
                )
                
                # Parse response
                result = json.loads(response.content[0].text)
                return result
    
    async def parse_claim(self, claim_text: str, use_llm: bool = True) -> Dict[str, Any]:
        """Parse a claim submission"""
        return await self.call_tool(
            AgentEndpoint.PARSE_CLAIM,
            {"text": claim_text, "use_llm": use_llm}
        )
    
    async def process_complete_claim(self, claim_text: str, customer_name: str = "") -> Dict[str, Any]:
        """Process a complete claim through all agents"""
        return await self.call_tool(
            AgentEndpoint.COMPLETE_PIPELINE,
            {"claim_text": claim_text, "customer_name": customer_name}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        return await self.call_tool(
            AgentEndpoint.HEALTH_CHECK,
            {}
        )


async def interactive_mode():
    """Interactive command line interface"""
    
    client = MCPInsuranceClient()
    
    print("=" * 60)
    print("MCP Insurance Claim Processing System")
    print("=" * 60)
    print("\nCommands:")
    print("  parse <text>     - Parse a claim submission")
    print("  process <text>   - Complete claim processing")
    print("  health           - Check system health")
    print("  quit             - Exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'health':
                result = await client.health_check()
                print("\nSystem Health:")
                print(json.dumps(result, indent=2))
                continue
            
            if user_input.startswith('parse '):
                claim_text = user_input[6:]
                print("\n📝 Parsing claim...")
                result = await client.parse_claim(claim_text)
                print("\nParsed Fields:")
                print(json.dumps(result.get('result', result), indent=2))
                continue
            
            if user_input.startswith('process '):
                claim_text = user_input[8:]
                print("\n🔄 Processing complete claim...")
                result = await client.process_complete_claim(claim_text)
                
                if 'error' in result:
                    print(f"\n❌ Error: {result['error']}")
                else:
                    print("\n✅ Claim Processed Successfully!")
                    print(f"Status: {result.get('status', 'UNKNOWN')}")
                    print(f"Final Decision: {result.get('final_decision', 'UNKNOWN')}")
                    print(f"Risk Score: {result.get('risk_assessment', {}).get('risk_score', 'N/A')}")
                    print(f"Fraud Probability: {result.get('fraud_prediction', {}).get('fraud_probability', 'N/A')}")
                    print(f"\nSummary:\n{result.get('summary', 'No summary available')}")
                continue
            
            print("Unknown command. Try: parse, process, health, or quit")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


async def batch_process(file_path: str):
    """Process multiple claims from a file"""
    
    client = MCPInsuranceClient()
    
    with open(file_path, 'r') as f:
        claims = [line.strip() for line in f if line.strip()]
    
    results = []
    
    for i, claim in enumerate(claims, 1):
        print(f"Processing claim {i}/{len(claims)}...")
        result = await client.process_complete_claim(claim)
        results.append(result)
    
    # Save results
    output_file = Path(file_path).stem + "_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Processed {len(claims)} claims. Results saved to {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Insurance Client")
    parser.add_argument("--batch", help="Batch process claims from file")
    parser.add_argument("--claim", help="Process a single claim")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.batch:
        asyncio.run(batch_process(args.batch))
    elif args.claim:
        async def single():
            client = MCPInsuranceClient()
            result = await client.process_complete_claim(args.claim)
            print(json.dumps(result, indent=2))
        asyncio.run(single())
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()