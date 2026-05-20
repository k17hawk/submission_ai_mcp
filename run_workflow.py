# run_claim.py
import asyncio
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lang_graph.workflow import graph
from src.lang_graph.state import ClaimProcessingState  

async def process_claim(claim_text: str, customer_name: str, customer_id: str):
    initial_state = {
        "claim_text": claim_text,
        "customer_name": customer_name,
        "customer_id": customer_id,
        "parsed_claim": None,
        "policy_verification": None,
        "risk_assessment": None,
        "feature_vector": None,
        "fraud_prediction": None,
        "final_decision": None,
        "recommended_action": None,
        "status": None,
        "missing_fields": [],
        "errors": [],
        "processing_time_ms": 0.0,
        "llm_enhanced": False,
        "extraction_success": False,
        "policy_found": False,
        "critical_missing": []
    }
    
    config = {"configurable": {"thread_id": customer_id}} 
    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state

def visualize_graph():
    """Generate and save graph diagrams."""
    print("📊 Generating LangGraph visualisation...")
    
    # ASCII
    print("\n=== ASCII Graph ===\n")
    print(graph.get_graph().draw_ascii())
    
    # Save PNG
    try:
        png_data = graph.get_graph().draw_png()
        with open("insurance_claim_graph.png", "wb") as f:
            f.write(png_data)
        print("\n✅ Graph saved as insurance_claim_graph.png")
    except Exception as e:
        print(f"⚠️ Could not generate PNG: {e}. Install playwright: pip install playwright && playwright install")
    
    # Optional: save Mermaid code
    mermaid_code = graph.get_graph().draw_mermaid()
    with open("insurance_claim_graph.mmd", "w") as f:
        f.write(mermaid_code)
    print("✅ Mermaid code saved as insurance_claim_graph.mmd")

if __name__ == "__main__":
    # First, visualise the graph
    visualize_graph()
    
    # Then run a sample claim
    sample_text = """Was involved in a 2-car pileup on I-95 at 2018-08-09 at FL Brownhaven and it has been 8 hours where i had Front Collision.
                    My  2011 Chevrolet Equinox. Policy POL-651065.
                    Name: Allison Hill with CUST-93810. 1 witnesses saw everything.
                    Police report available. Damage around $8,000."""
    
    print("\n=== Processing Claim ===\n")
    result = asyncio.run(process_claim(sample_text, "Allison Hill", "CUST-93810"))
    print("\n=== Result ===")
    print("Final decision:", result["final_decision"])
    print("Recommended action:", result["recommended_action"])
    print("Status:", result["status"])
    if result.get("errors"):
        print("Errors:", result["errors"])