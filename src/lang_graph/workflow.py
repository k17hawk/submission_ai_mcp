
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 
from src.lang_graph.state import ClaimProcessingState
from src.lang_graph.node import (
    claim_parsing_agent, feature_building_agent, final_decision_agent, fraud_detection_agent, fraud_detection_agent,
     policy_lookup_agent, reject_claim_agent, risk_assessment_agent
)

def route_after_parsing(state: ClaimProcessingState) -> str:
    if not state.get("extraction_success") or not state.get("parsed_claim"):
        return "reject"
    return "continue"

def route_after_policy(state: ClaimProcessingState) -> str:
    if not state.get("policy_found"):
        return "reject"
    return "continue"

def route_after_risk(state: ClaimProcessingState) -> str:
    # If risk score already auto‑denies, we could skip fraud? But keep for completeness.
    return "continue"

# Build the graph
builder = StateGraph(ClaimProcessingState)

# Add nodes
builder.add_node("parse", claim_parsing_agent)
builder.add_node("policy", policy_lookup_agent)
builder.add_node("risk", risk_assessment_agent)
builder.add_node("features", feature_building_agent)
builder.add_node("fraud", fraud_detection_agent)
builder.add_node("decision", final_decision_agent)
builder.add_node("reject", reject_claim_agent   )

#entry point
builder.set_entry_point("parse")

# Conditional edges
builder.add_conditional_edges(
    "parse",
    route_after_parsing,
    {
        "continue": "policy",
        "reject": "reject"
    }
)

builder.add_conditional_edges(
    "policy",
    route_after_policy,
    {
        "continue": "risk",
        "reject": "reject"
    }
)
builder.add_edge("risk", "features")
builder.add_edge("features", "fraud")
builder.add_edge("fraud", "decision")

builder.add_edge("decision", END)
builder.add_edge("reject", END)

graph = builder.compile(checkpointer=MemorySaver())