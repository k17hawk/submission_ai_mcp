# underwriting_graph.py
"""
Multi-Agent Underwriting Assistant using LangGraph + MCP Servers
"""

import asyncio
import logging
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from mcp_client import mcp_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#state definition for the underwriting process
class UnderwritingState(TypedDict):
    # Input
    pdf_path: str
    submission_id: str
    
    # Agent 1 output
    full_text: str
    policy_data: Dict[str, Any]
    parse_error: Optional[str]
    
    # Agent 2 output
    detected_categories: List[str]
    search_results: Dict[str, List[Dict]]
    
    # Agent 3 output (Using Risk Server instead of Insurance Server)
    risk_assessment: Dict[str, Any]
    clause_analyses: List[Dict[str, Any]]
    
    # Agent 4 output
    recommendations: List[Dict[str, Any]]
    negotiation_points: List[Dict[str, Any]]
    strong_points: List[Dict[str, Any]]
    final_decision: str
    decision_emoji: str
    
    # Agent 5 output
    executive_summary: str
    full_report: str
    report_generated_at: str
    
    # Logs
    agent_logs: List[str]
    errors: List[str]


# ═══════════════════════════════════════════════════════════════
# AGENT 1: POLICY EXTRACTOR (Insurance Server)
# ═══════════════════════════════════════════════════════════════

async def extractor_agent(state: UnderwritingState) -> UnderwritingState:
    """Parse PDF using Insurance MCP Server"""
    logger.info("🔍 AGENT 1: EXTRACTOR — Parsing PDF...")
    state["agent_logs"].append("🔍 EXTRACTOR: Starting PDF analysis via Insurance Server...")
    
    try:
        result = await mcp_manager.parse_submission(state["pdf_path"])
        
        if isinstance(result, dict) and result.get("error"):
            state["parse_error"] = result["error"]
            state["errors"].append(f"Parse error: {result['error']}")
            state["agent_logs"].append(f"❌ EXTRACTOR: Failed — {result['error']}")
            return state
        
        # Handle both dict and string responses
        if isinstance(result, dict):
            state["full_text"] = result.get("text", "")
            state["policy_data"] = result.get("policy_data", {})
        else:
            state["full_text"] = str(result)
            state["policy_data"] = {}
        
        policy_type = state["policy_data"].get("policy_type", "Unknown")
        insured = state["policy_data"].get("insured_name", "N/A")
        policy_num = state["policy_data"].get("policy_number", "N/A")
        
        state["agent_logs"].append(
            f"✅ EXTRACTOR: {policy_type} | {insured} | {policy_num}"
        )
        state["agent_logs"].append(
            f"   Extracted {len(state['full_text'])} characters"
        )
        
    except Exception as e:
        state["parse_error"] = str(e)
        state["errors"].append(f"Extractor error: {e}")
        state["agent_logs"].append(f"❌ EXTRACTOR: Exception — {e}")
    
    return state


#clause analyzer agent using insurance server for category detection and corpus search
async def analyzer_agent(state: UnderwritingState) -> UnderwritingState:
    """Detect clauses and search corpus using Insurance Server"""
    logger.info("🔍 AGENT 2: ANALYZER — Detecting clauses...")
    state["agent_logs"].append("🔍 ANALYZER: Detecting clause types via Insurance Server...")
    
    if state.get("parse_error"):
        state["agent_logs"].append("⚠️ ANALYZER: Skipped — parser failed")
        return state
    
    try:
        # Try to auto-detect categories
        categories = await mcp_manager.detect_categories(state["full_text"])
        state["detected_categories"] = categories
        
        if not categories:
            # Fallback: use common categories
            categories = [
                "cap on liability",
                "indemnification",
                "warranty",
                "termination",
                "confidentiality"
            ]
            state["detected_categories"] = categories
            state["agent_logs"].append("⚠️ Using default categories")
        
        state["agent_logs"].append(
            f"✅ ANALYZER: Found {len(categories)} clause types"
        )
        
        # Search corpus for each category
        search_results = {}
        for cat in categories:
            try:
                results = await mcp_manager.search_corpus(cat, top_k=3)
                search_results[cat] = results if isinstance(results, list) else []
                state["agent_logs"].append(
                    f"   📄 '{cat}': {len(search_results[cat])} matches"
                )
            except Exception as e:
                state["agent_logs"].append(f"   ⚠️ '{cat}': search failed — {e}")
                search_results[cat] = []
        
        state["search_results"] = search_results
        
    except Exception as e:
        state["agent_logs"].append(f"❌ ANALYZER: Failed — {e}")
        state["errors"].append(f"Analyzer error: {e}")
    
    return state


#agent 3: risk assessor using Risk Server
async def risk_assessor_agent(state: UnderwritingState) -> UnderwritingState:
    """Assess risk using Risk Server (analyze_submission_text)"""
    logger.info("🔍 AGENT 3: RISK ASSESSOR — Using Risk Server...")
    state["agent_logs"].append("📊 RISK ASSESSOR: Analyzing risk via Risk Server...")
    
    if not state.get("full_text"):
        state["agent_logs"].append("⚠️ RISK ASSESSOR: No text to analyze")
        return state
    
    try:
        policy_type = state.get("policy_data", {}).get("policy_type")
        
        # Use Risk Server's analyze_submission_text (NOT insurance server's assess_submission_risk)
        risk_result = await mcp_manager.analyze_submission_risk(
            full_text=state["full_text"],
            policy_type=policy_type
        )
        
        state["risk_assessment"] = risk_result
        
        # Extract clause-level analyses
        clause_analyses = []
        if isinstance(risk_result, dict):
            # Handle different response structures
            clauses = risk_result.get("clauses", [])
            if not clauses and "results" in risk_result:
                clauses = risk_result.get("results", [])
            
            for clause in clauses:
                if isinstance(clause, dict):
                    clause_analyses.append({
                        "clause_text": clause.get("text", clause.get("clause_text", "")),
                        "rating": clause.get("rating", clause.get("predicted_rating", "N/A")),
                        "risk_level": clause.get("risk_level", clause.get("overall_risk", "N/A")),
                        "stars": clause.get("stars", "☆☆☆☆☆"),
                    })
        
        state["clause_analyses"] = clause_analyses
        
        # Log results
        overall_risk = risk_result.get("overall_risk", "UNKNOWN") if isinstance(risk_result, dict) else "UNKNOWN"
        avg_rating = risk_result.get("average_rating", 0) if isinstance(risk_result, dict) else 0
        
        state["agent_logs"].append(
            f"✅ RISK ASSESSOR: Overall Risk = {overall_risk} | Avg Rating = {avg_rating}/5.0"
        )
        state["agent_logs"].append(
            f"   Analyzed {len(clause_analyses)} clauses"
        )
        
    except Exception as e:
        state["agent_logs"].append(f"❌ Risk assessment failed: {e}")
        state["errors"].append(f"Risk assessor error: {e}")
        state["risk_assessment"] = {"error": str(e)}
    
    return state


#underwriting advisor agent that makes recommendations based on risk assessment and clause analyses
async def advisor_agent(state: UnderwritingState) -> UnderwritingState:
    """Make underwriting recommendations based on risk analysis"""
    logger.info("🔍 AGENT 4: ADVISOR — Making recommendations...")
    state["agent_logs"].append("🎯 ADVISOR: Formulating recommendations...")
    
    risk = state.get("risk_assessment", {})
    clauses = state.get("clause_analyses", [])
    
    if not risk or risk.get("error"):
        state["final_decision"] = "ERROR — Risk assessment failed"
        state["decision_emoji"] = "❌"
        state["agent_logs"].append("❌ ADVISOR: Cannot recommend — no risk data")
        return state
    
    strong_points = []
    negotiation_points = []
    
    # Analyze each clause
    for clause in clauses:
        rating = clause.get("rating", 0)
        clause_text = clause.get("clause_text", "")[:100]
        stars = clause.get("stars", "☆☆☆☆☆")
        
        try:
            rating_float = float(rating) if rating != "N/A" else 0
        except (ValueError, TypeError):
            rating_float = 0
        
        if rating_float >= 4.0:
            strong_points.append({
                "clause": clause_text,
                "rating": rating_float,
                "stars": stars,
                "action": "No change needed — strong protection"
            })
        elif rating_float >= 2.5:
            negotiation_points.append({
                "clause": clause_text,
                "rating": rating_float,
                "stars": stars,
                "action": "Consider strengthening",
                "priority": "SHOULD"
            })
        else:
            negotiation_points.append({
                "clause": clause_text,
                "rating": rating_float,
                "stars": stars,
                "action": "Must be revised",
                "priority": "MUST"
            })
    
    # Check for missing protections
    if isinstance(risk, dict):
        missing = risk.get("missing_protections", [])
        for m in missing:
            negotiation_points.append({
                "clause": m if isinstance(m, str) else m.get("category", "Unknown"),
                "rating": 0,
                "stars": "☆☆☆☆☆",
                "action": "Add missing protection",
                "priority": "MUST"
            })
    
    # Sort by priority
    priority_order = {"MUST": 0, "SHOULD": 1}
    negotiation_points.sort(key=lambda x: priority_order.get(x.get("priority", "SHOULD"), 1))
    
    state["strong_points"] = strong_points
    state["negotiation_points"] = negotiation_points
    
    # Make decision
    avg_rating = risk.get("average_rating", 0) if isinstance(risk, dict) else 0
    must_count = sum(1 for n in negotiation_points if n["priority"] == "MUST")
    
    if avg_rating >= 4.0 and must_count == 0:
        decision = "ACCEPT — Strong policy"
        emoji = "✅"
    elif avg_rating >= 3.0 and must_count <= 1:
        decision = "ACCEPT WITH CONDITIONS — Minor revisions needed"
        emoji = "⚠️"
    elif avg_rating >= 2.5 and must_count <= 3:
        decision = "REFER TO SENIOR UNDERWRITER — Significant concerns"
        emoji = "🔶"
    else:
        decision = "REJECT — Major risk factors present"
        emoji = "🔴"
    
    state["final_decision"] = decision
    state["decision_emoji"] = emoji
    state["agent_logs"].append(f"🎯 ADVISOR: {emoji} {decision}")
    
    return state


#reporter agent that generates a final report based on all previous analyses and recommendations
async def reporter_agent(state: UnderwritingState) -> UnderwritingState:
    """Generate final report"""
    logger.info("🔍 AGENT 5: REPORTER — Generating report...")
    state["agent_logs"].append("📋 REPORTER: Generating final report...")
    
    policy = state.get("policy_data", {})
    risk = state.get("risk_assessment", {})
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["report_generated_at"] = now
    
    # Create executive summary
    lines = [
        "=" * 60,
        "     MULTI-AGENT UNDERWRITING REPORT",
        "=" * 60,
        f"\n📄 POLICY: {policy.get('policy_number', 'N/A')}",
        f"🏢 INSURED: {policy.get('insured_name', 'N/A')}",
        f"📋 TYPE: {policy.get('policy_type', 'N/A')}",
        f"📅 PERIOD: {policy.get('effective_date', 'N/A')} → {policy.get('expiration_date', 'N/A')}",
        f"\n📊 OVERALL RISK: {risk.get('overall_risk', 'N/A') if isinstance(risk, dict) else 'N/A'}",
        f"⭐ AVERAGE RATING: {risk.get('average_rating', 'N/A') if isinstance(risk, dict) else 'N/A'}/5.0",
        f"\n🎯 DECISION: {state.get('decision_emoji', '')} {state.get('final_decision', 'N/A')}",
    ]
    
    # Strong points
    if state.get("strong_points"):
        lines.append("\n🟢 STRONG POINTS:")
        for sp in state["strong_points"][:3]:
            lines.append(f"   ✅ {sp['clause'][:80]}... ({sp['rating']}★)")
    
    # Required actions
    if state.get("negotiation_points"):
        lines.append("\n🔴 REQUIRED ACTIONS:")
        for np in state["negotiation_points"][:5]:
            priority_tag = "MUST" if np["priority"] == "MUST" else "SHOULD"
            emoji = "🔴" if priority_tag == "MUST" else "🟡"
            lines.append(f"   {emoji} [{priority_tag}] {np['action']}")
    
    lines.append(f"\n📅 Report generated: {now}")
    lines.append(f"\n🤖 Pipeline: Insurance Server → Risk Server → Advisor → Report")
    
    state["executive_summary"] = "\n".join(lines)
    state["full_report"] = state["executive_summary"] + f"\n\nAgent Logs:\n" + "\n".join(state["agent_logs"])
    
    state["agent_logs"].append("✅ REPORTER: Report generated successfully")
    return state


#buid graph that connects all agents in sequence and compiles it with memory checkpointing
def build_underwriting_graph():
    """Build and compile the LangGraph workflow"""
    
    workflow = StateGraph(UnderwritingState)
    
    # Add nodes
    workflow.add_node("extractor", extractor_agent)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("risk_assessor", risk_assessor_agent)
    workflow.add_node("advisor", advisor_agent)
    workflow.add_node("reporter", reporter_agent)
    
    # Define edges
    workflow.add_edge("extractor", "analyzer")
    workflow.add_edge("analyzer", "risk_assessor")
    workflow.add_edge("risk_assessor", "advisor")
    workflow.add_edge("advisor", "reporter")
    workflow.add_edge("reporter", END)
    
    workflow.set_entry_point("extractor")
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


#pipeline running
async def run_underwriting_pipeline(
    pdf_path: str,
    submission_id: str = "default"
) -> Dict[str, Any]:
    """Run the complete underwriting pipeline"""
    
    app = build_underwriting_graph()
    
    initial_state: UnderwritingState = {
        "pdf_path": pdf_path,
        "submission_id": submission_id,
        "full_text": "",
        "policy_data": {},
        "parse_error": None,
        "detected_categories": [],
        "search_results": {},
        "risk_assessment": {},
        "clause_analyses": [],
        "recommendations": [],
        "negotiation_points": [],
        "strong_points": [],
        "final_decision": "",
        "decision_emoji": "",
        "executive_summary": "",
        "full_report": "",
        "report_generated_at": "",
        "agent_logs": [],
        "errors": [],
    }
    
    config = {"configurable": {"thread_id": submission_id}}
    
    result = await app.ainvoke(initial_state, config)
    return result