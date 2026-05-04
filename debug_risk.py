#!/usr/bin/env python3
"""Debug what the risk server actually returns"""

import asyncio
import json
from mcp_client import mcp_manager

async def debug_risk_server():
    print("=" * 70)
    print("DEBUGGING RISK SERVER RESPONSE")
    print("=" * 70)
    
    # Sample text that should trigger ratings
    sample_text = """
    CAP ON LIABILITY: The total liability of the Company under this policy shall not exceed $1,000,000 per occurrence and $2,000,000 in the aggregate.
    
    INDEMNIFICATION: The Insured shall indemnify and hold harmless the Company against any and all claims arising from the Insured's negligence.
    
    TERMINATION: Either party may terminate this agreement with 30 days written notice.
    
    WARRANTY: The Insured warrants that all statements in the application are true and complete.
    """
    
    print("\n📝 Testing with sample text...")
    
    try:
        result = await mcp_manager.analyze_submission_risk(
            full_text=sample_text,
            policy_type="commercial_general_liability"
        )
        
        print("\n📊 RAW RESULT FROM RISK SERVER:")
        print("-" * 70)
        print(json.dumps(result, indent=2, default=str)[:2000])
        print("-" * 70)
        
        print("\n🔍 RESULT STRUCTURE ANALYSIS:")
        print(f"Type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"Keys: {list(result.keys())}")
            
            # Check for rating-related fields
            rating_fields = ['rating', 'ratings', 'average_rating', 'avg_rating', 'overall_risk', 'risk_level']
            for field in rating_fields:
                if field in result:
                    print(f"✓ Found '{field}': {result[field]}")
            
            # Check for clauses
            clause_fields = ['clauses', 'results', 'clause_analyses', 'analyzed_clauses']
            for field in clause_fields:
                if field in result:
                    clauses = result[field]
                    print(f"✓ Found '{field}': {len(clauses) if isinstance(clauses, list) else 'not a list'}")
                    
                    if isinstance(clauses, list) and len(clauses) > 0:
                        print(f"  First clause example:")
                        first_clause = clauses[0]
                        if isinstance(first_clause, dict):
                            for k, v in list(first_clause.items())[:5]:
                                print(f"    {k}: {v}")
        
        elif isinstance(result, list):
            print(f"Result is a list with {len(result)} items")
            if len(result) > 0:
                print(f"First item: {json.dumps(result[0], indent=2)[:300]}")
        
        else:
            print(f"Result is {type(result)}: {str(result)[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_risk_server())