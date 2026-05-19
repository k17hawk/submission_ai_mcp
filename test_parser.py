import asyncio
from src.mcp_submission_parsing.servers.agent_parser import ParserAgent

async def test():
    claim_text = """Was involved in a 2-car pileup on I-95 at 2018-08-09 at FL Brownhaven and it has been 8 hours where i had Front Collision.
    My  2011 Chevrolet Equinox. Policy POL-651065.
    Name: Allison Hill with CUST-93810. 1 witnesses saw everything.
    Police report available. Damage around $8,000."""

    parser = ParserAgent()
    
    # Test 1: Normal mode - LLM only called if fields are missing
    print("\n=== TEST 1: Normal Mode (LLM only if missing) ===")
    result1 = await parser({"text": claim_text, "use_llm": True})
    print(f"LLM Enhanced: {result1.get('llm_enhanced')}")
    print(f"Missing fields: {result1.get('missing_fields')}")
    print(f"Result: {result1}\n")
    
    # Test 2: Force LLM enhancement - LLM called even when all fields present
    print("\n=== TEST 2: Force LLM Mode (always call LLM) ===")
    result2 = await parser({
        "text": claim_text, 
        "use_llm": False,
        "force_llm_enhancement": False  
    })
    print(f"LLM Enhanced: {result2.get('llm_enhanced')}")
    print(f"LLM Filled Count: {result2.get('llm_filled_count')}")
    print(f"LLM Overrode Count: {result2.get('llm_overrode_count')}")
    print(f"Missing fields: {result2.get('missing_fields')}")
    print(f"Result: {result2}")

if __name__ == "__main__":
    asyncio.run(test())