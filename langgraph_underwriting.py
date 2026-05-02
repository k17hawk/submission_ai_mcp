"""
Multi-Agent Underwriting Assistant
Orchestrates Submission Parser (MCP Server 1) + Risk Checker (MCP Server 2)
using LangGraph.
"""

import asyncio
import logging
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─── MCP Client imports ───
# Your two servers are at:
#   Server 1 (Parser): http://127.0.0.1:8000
#   Server 2 (Risk):   http://127.0.0.1:8001
# In production, you'd use MCP client libraries. For now, we call the tools directly.

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp_insurance.tools.parsing import parse_acord_submission, extract_policy_data
from src.mcp_insurance.tools.retrieval import search_corpus, get_document_by_id
from src.mcp_insurance.tools.rating import rate_clause, get_rating_examples, get_available_categories
from src.mcp_insurance.tools.risk import assess_submission_risk, _make_stars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════

class UnderwritingState(TypedDict):
    # Input
    pdf_path: str
    
    # Agent 1 output
    full_text: str
    policy_data: Dict[str, Any]
    parse_error: Optional[str]
    
    # Agent 2 output
    detected_categories: List[str]
    search_results: Dict[str, List[Dict]]
    
    # Agent 3 output
    rated_clauses: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    
    # Agent 4 output
    recommendations: List[Dict[str, Any]]
    negotiation_points: List[Dict[str, Any]]
    strong_points: List[Dict[str, Any]]
    final_decision: str
    
    # Agent 5 output
    executive_summary: str
    full_report: str
    
    # Logs
    agent_logs: List[str]
    errors: List[str]


# ═══════════════════════════════════════════════════════════════
# AGENT 1: POLICY EXTRACTOR
# ═══════════════════════════════════════════════════════════════

async def extractor_agent(state: UnderwritingState) -> UnderwritingState:
    """
    Reads the PDF and extracts policy fields and full text.
    Calls: MCP Server 1 → parse_acord_submission()
    """
    logger.info("🔍 AGENT 1: EXTRACTOR — Reading PDF...")
    state["agent_logs"].append("🔍 EXTRACTOR: Starting PDF analysis...")
    
    try:
        result = await parse_acord_submission(state["pdf_path"])
        
        if result.get("error"):
            state["parse_error"] = result["error"]
            state["errors"].append(f"Parse error: {result['error']}")
            state["agent_logs"].append(f"❌ EXTRACTOR: Failed — {result['error']}")
            return state
        
        state["full_text"] = result.get("text", "")
        state["policy_data"] = result.get("policy_data", {})
        
        policy_type = state["policy_data"].get("policy_type", "Unknown")
        insured = state["policy_data"].get("insured_name", "Unknown")
        policy_num = state["policy_data"].get("policy_number", "Unknown")
        
        state["agent_logs"].append(
            f"✅ EXTRACTOR: {policy_type} | {insured} | {policy_num}"
        )
        state["agent_logs"].append(
            f"   Extracted {len(state['full_text'])} characters of text"
        )
        
    except Exception as e:
        state["parse_error"] = str(e)
        state["errors"].append(f"Extractor error: {e}")
        state["agent_logs"].append(f"❌ EXTRACTOR: Exception — {e}")
    
    return state


# ═══════════════════════════════════════════════════════════════
# AGENT 2: CLAUSE ANALYZER
# ═══════════════════════════════════════════════════════════════

async def analyzer_agent(state: UnderwritingState) -> UnderwritingState:
    """
    Detects clause categories from the policy text.
    Calls: MCP Server 1 → search_corpus() to find matching clauses
    """
    logger.info("🔍 AGENT 2: ANALYZER — Detecting clauses...")
    state["agent_logs"].append("🔍 ANALYZER: Searching for clause types...")
    
    if state.get("parse_error"):
        state["agent_logs"].append("⚠️ ANALYZER: Skipped — parser failed")
        return state
    
    from src.mcp_insurance.tools.pipeline import _detect_categories_from_text
    
    categories = _detect_categories_from_text(state["full_text"])
    state["detected_categories"] = categories
    
    if not categories:
        state["agent_logs"].append("⚠️ ANALYZER: No clause categories detected")
        return state
    
    state["agent_logs"].append(
        f"✅ ANALYZER: Found {len(categories)} clause types"
    )
    
    # For each category, search the corpus for similar clauses
    search_results = {}
    for cat in categories:
        try:
            results = await search_corpus(cat, top_k=3)
            search_results[cat] = results
            state["agent_logs"].append(f"   📄 '{cat}': {len(results)} matches found")
        except Exception as e:
            state["agent_logs"].append(f"   ⚠️ '{cat}': search failed — {e}")
            search_results[cat] = []
    
    state["search_results"] = search_results
    return state


# ═══════════════════════════════════════════════════════════════
# AGENT 3: RISK ASSESSOR
# ═══════════════════════════════════════════════════════════════

async def risk_assessor_agent(state: UnderwritingState) -> UnderwritingState:
    """
    Rates each detected clause and produces risk assessment.
    Calls: MCP Server 2 → rate_clause() + assess_submission_risk()
    """
    logger.info("🔍 AGENT 3: RISK ASSESSOR — Rating clauses...")
    state["agent_logs"].append("📊 RISK ASSESSOR: Analyzing risk profile...")
    
    if not state.get("detected_categories"):
        state["agent_logs"].append("⚠️ RISK ASSESSOR: No categories to rate")
        return state
    
    rated_clauses = []
    
    for category in state["detected_categories"]:
        # Use the top search result text for rating
        search_results = state.get("search_results", {}).get(category, [])
        
        if search_results:
            clause_text = search_results[0].get("text", state["full_text"])
        else:
            clause_text = state["full_text"]
        
        try:
            rating_result = await rate_clause(clause_text, category=category)
            
            rated_clauses.append({
                "category": category,
                "predicted_rating": rating_result.get("predicted_rating"),
                "stars": rating_result.get("stars", "☆☆☆☆☆"),
                "category_used": rating_result.get("category_used"),
                "best_match_id": search_results[0]["doc_id"] if search_results else None,
            })
            
            emoji = _get_emoji(rating_result.get("predicted_rating"))
            state["agent_logs"].append(
                f"   {emoji} {category}: {rating_result.get('predicted_rating', '?')}★ "
                f"{rating_result.get('stars', '')}"
            )
            
        except Exception as e:
            state["agent_logs"].append(f"   ❌ {category}: rating failed — {e}")
            rated_clauses.append({
                "category": category,
                "predicted_rating": None,
                "stars": "☆☆☆☆☆",
                "error": str(e),
            })
    
    state["rated_clauses"] = rated_clauses
    
    # ─── Full risk assessment ───
    policy_type = state.get("policy_data", {}).get("policy_type")
    
    try:
        mini_report = {
            "results": [
                {
                    "query": c["category"],
                    "rated_clauses": [
                        {
                            "predicted_rating": c["predicted_rating"],
                            "stars": c["stars"],
                        }
                    ]
                }
                for c in rated_clauses
            ]
        }
        risk = await assess_submission_risk(mini_report, policy_type=policy_type)
        state["risk_assessment"] = risk
        
        state["agent_logs"].append(
            f"📊 OVERALL RISK: {risk.get('overall_risk_emoji', '')} "
            f"{risk.get('overall_risk', 'UNKNOWN')} "
            f"({risk.get('average_rating', 0)}/5.0)"
        )
        
    except Exception as e:
        state["agent_logs"].append(f"❌ Risk assessment failed: {e}")
        state["risk_assessment"] = {"error": str(e)}
    
    return state


def _get_emoji(rating) -> str:
    """Return emoji based on rating."""
    if rating is None:
        return "⚪"
    if rating >= 4.0:
        return "🟢"
    elif rating >= 2.5:
        return "🟡"
    return "🔴"


# ═══════════════════════════════════════════════════════════════
# AGENT 4: UNDERWRITING ADVISOR
# ═══════════════════════════════════════════════════════════════

async def advisor_agent(state: UnderwritingState) -> UnderwritingState:
    """
    Makes underwriting recommendations based on risk analysis.
    Uses rules + risk thresholds to decide: ACCEPT / CONDITIONAL / REFER.
    """
    logger.info("🔍 AGENT 4: ADVISOR — Making recommendations...")
    state["agent_logs"].append("🎯 ADVISOR: Formulating recommendations...")
    
    risk = state.get("risk_assessment", {})
    rated = state.get("rated_clauses", [])
    
    if not rated:
        state["final_decision"] = "ERROR — No clauses rated"
        state["agent_logs"].append("❌ ADVISOR: Cannot recommend — no data")
        return state
    
    recommendations = []
    negotiation_points = []
    strong_points = []
    
    # ─── Analyze each clause ───
    for clause in rated:
        rating = clause.get("predicted_rating")
        category = clause.get("category", "unknown")
        stars = clause.get("stars", "☆☆☆☆☆")
        
        if rating is None:
            continue
        
        if rating >= 4.0:
            strong_points.append({
                "category": category,
                "rating": rating,
                "stars": stars,
                "action": "No change needed — strong protection"
            })
        elif rating >= 2.5:
            negotiation_points.append({
                "category": category,
                "rating": rating,
                "stars": stars,
                "action": "Consider strengthening",
                "priority": "SHOULD"
            })
        else:
            negotiation_points.append({
                "category": category,
                "rating": rating,
                "stars": stars,
                "action": "Must be revised",
                "priority": "MUST"
            })
    
    # ─── Check missing protections ───
    missing = risk.get("missing_protections", [])
    for m in missing:
        negotiation_points.append({
            "category": m.get("category", "unknown"),
            "rating": None,
            "stars": "☆☆☆☆☆",
            "action": "Add missing protection",
            "priority": "MUST"
        })
    
    # ─── Sort by priority ───
    priority_order = {"MUST": 0, "SHOULD": 1, "NICE_TO_HAVE": 2}
    negotiation_points.sort(key=lambda x: priority_order.get(x.get("priority", "NICE_TO_HAVE"), 3))
    
    state["recommendations"] = recommendations
    state["negotiation_points"] = negotiation_points
    state["strong_points"] = strong_points
    
    # ─── Final Decision ───
    avg_rating = risk.get("average_rating", 0)
    high_count = risk.get("high_risk_count", 0)
    must_count = sum(1 for n in negotiation_points if n["priority"] == "MUST")
    
    if avg_rating >= 4.0 and must_count == 0:
        decision = "✅ ACCEPT — Strong policy"
    elif avg_rating >= 3.0 and must_count <= 1:
        decision = "⚠️ ACCEPT WITH CONDITIONS — Minor revisions needed"
    elif avg_rating >= 2.5 and must_count <= 3:
        decision = "🔶 REFER TO SENIOR UNDERWRITER — Significant concerns"
    else:
        decision = "🔴 REJECT — Major risk factors present"
    
    state["final_decision"] = decision
    state["agent_logs"].append(f"🎯 ADVISOR: {decision}")
    
    return state


# ═══════════════════════════════════════════════════════════════
# AGENT 5: REPORTER
# ═══════════════════════════════════════════════════════════════

async def reporter_agent(state: UnderwritingState) -> UnderwritingState:
    """
    Formats the final executive summary and full report.
    """
    logger.info("🔍 AGENT 5: REPORTER — Generating report...")
    state["agent_logs"].append("📋 REPORTER: Generating final report...")
    
    policy = state.get("policy_data", {})
    risk = state.get("risk_assessment", {})
    
    # ─── Executive Summary ───
    summary_lines = [
        "╔══════════════════════════════════════════════════╗",
        "║     MULTI-AGENT UNDERWRITING REPORT             ║",
        "╚══════════════════════════════════════════════════╝",
        "",
        f"📄 POLICY: {policy.get('policy_number', 'N/A')}",
        f"🏢 INSURED: {policy.get('insured_name', 'N/A')}",
        f"📋 TYPE: {policy.get('policy_type', 'N/A')}",
        f"📅 PERIOD: {policy.get('effective_date', 'N/A')} → {policy.get('expiration_date', 'N/A')}",
        "",
        f"📊 OVERALL RISK: {risk.get('overall_risk_emoji', '')} {risk.get('overall_risk', 'N/A')}",
        f"⭐ AVERAGE RATING: {risk.get('average_rating', 'N/A')}/5.0 {risk.get('average_stars', '')}",
        "",
        f"🎯 DECISION: {state.get('final_decision', 'N/A')}",
        "",
    ]
    
    # Strong points
    if state.get("strong_points"):
        summary_lines.append("🟢 STRONG POINTS:")
        for sp in state["strong_points"]:
            summary_lines.append(f"   ✅ {sp['category']}: {sp['rating']}★ {sp['stars']}")
        summary_lines.append("")
    
    # Required actions
    if state.get("negotiation_points"):
        summary_lines.append("🔴 REQUIRED ACTIONS:")
        for np in state["negotiation_points"]:
            tag = "MUST" if np["priority"] == "MUST" else "SHOULD"
            emoji = "🔴" if tag == "MUST" else "🟡"
            summary_lines.append(f"   {emoji} [{tag}] {np['category']}: {np['action']}")
        summary_lines.append("")
    
    # Agent processing summary
    summary_lines.append("🤖 AGENT PROCESSING SUMMARY:")
    for log in state.get("agent_logs", []):
        summary_lines.append(f"   {log}")
    
    state["executive_summary"] = "\n".join(summary_lines)
    state["full_report"] = state["executive_summary"]  # Could be extended
    
    state["agent_logs"].append("✅ REPORTER: Report generated")
    return state


# ═══════════════════════════════════════════════════════════════
# CONDITIONAL ROUTING
# ═══════════════════════════════════════════════════════════════

def should_continue(state: UnderwritingState) -> str:
    """Determine if pipeline should continue or abort."""
    if state.get("parse_error"):
        logger.warning("Parse error detected — routing to reporter with error")
        return "reporter"  # Skip analysis, go straight to report
    if not state.get("detected_categories"):
        logger.warning("No categories detected — routing to reporter")
        return "reporter"
    return "continue"


# ═══════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ═══════════════════════════════════════════════════════════════

def build_underwriting_graph():
    """Build and compile the LangGraph workflow."""
    
    workflow = StateGraph(UnderwritingState)
    
    # Add nodes
    workflow.add_node("extractor", extractor_agent)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("risk_assessor", risk_assessor_agent)
    workflow.add_node("advisor", advisor_agent)
    workflow.add_node("reporter", reporter_agent)
    
    # Happy path: extractor → analyzer → risk → advisor → reporter → END
    workflow.add_edge("extractor", "analyzer")
    workflow.add_edge("analyzer", "risk_assessor")
    workflow.add_edge("risk_assessor", "advisor")
    workflow.add_edge("advisor", "reporter")
    workflow.add_edge("reporter", END)
    
    # Entry point
    workflow.set_entry_point("extractor")
    
    # Compile with memory (enables pause/resume for human-in-the-loop)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

async def run_underwriting_assistant(pdf_path: str, thread_id: str = "default") -> Dict[str, Any]:
    """
    Run the complete multi-agent underwriting pipeline.
    
    Args:
        pdf_path: Path to the insurance submission PDF.
        thread_id: Unique ID for this run (enables conversation memory).
    
    Returns:
        Final state with all agent outputs.
    """
    app = build_underwriting_graph()
    
    initial_state: UnderwritingState = {
        "pdf_path": pdf_path,
        "full_text": "",
        "policy_data": {},
        "parse_error": None,
        "detected_categories": [],
        "search_results": {},
        "rated_clauses": [],
        "risk_assessment": {},
        "recommendations": [],
        "negotiation_points": [],
        "strong_points": [],
        "final_decision": "",
        "executive_summary": "",
        "full_report": "",
        "agent_logs": [],
        "errors": [],
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = await app.ainvoke(initial_state, config)
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    async def main():
        import sys
        
        if len(sys.argv) < 2:
            print("Usage: python langgraph_underwriting.py <path_to_pdf>")
            sys.exit(1)
        
        pdf_path = sys.argv[1]
        
        print("🚀 Starting Multi-Agent Underwriting Assistant...")
        print(f"📄 Processing: {pdf_path}")
        print("-" * 60)
        
        result = await run_underwriting_assistant(pdf_path)
        
        # Print executive summary
        print(result["executive_summary"])
        print("-" * 60)
        print(f"🤖 Total agent steps: {len(result['agent_logs'])}")
        
    asyncio.run(main())