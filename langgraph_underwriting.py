# underwriting_graph.py
"""
Multi-Agent Underwriting Assistant using LangGraph + MCP Servers
"""

import asyncio
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from mcp_client import mcp_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper functions
def safe_get(data: Any, key: str, default: Any = None) -> Any:
    """Safely get a key from dict or return default"""
    if isinstance(data, dict):
        return data.get(key, default)
    if hasattr(data, key):
        return getattr(data, key, default)
    return default

def ensure_dict(data: Any) -> dict:
    """Ensure data is converted to a dictionary"""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return {"text": data}
    try:
        return dict(data)
    except:
        return {"data": str(data)}


# ═══════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════

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
    
    # Agent 3 output (Using Risk Server)
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
# AGENT 1: POLICY EXTRACTOR
# ═══════════════════════════════════════════════════════════════

async def extractor_agent(state: UnderwritingState) -> UnderwritingState:
    """Parse PDF using Insurance MCP Server"""
    logger.info("🔍 AGENT 1: EXTRACTOR — Parsing PDF...")
    state["agent_logs"].append("🔍 EXTRACTOR: Starting PDF analysis via Insurance Server...")
    
    try:
        result = await mcp_manager.parse_submission(state["pdf_path"])
        result = ensure_dict(result)
        
        if result.get("error"):
            state["parse_error"] = result["error"]
            state["errors"].append(f"Parse error: {result['error']}")
            state["agent_logs"].append(f"❌ EXTRACTOR: Failed — {result['error']}")
            return state
        
        # Extract text and policy data
        state["full_text"] = result.get("text", "")
        state["policy_data"] = result.get("policy_data", {}) or {}
        
        policy_type = safe_get(state["policy_data"], "policy_type", "Unknown")
        insured = safe_get(state["policy_data"], "insured_name", "N/A")
        policy_num = safe_get(state["policy_data"], "policy_number", "N/A")
        
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
        logger.error(f"Extractor error: {e}", exc_info=True)
    
    return state


# ═══════════════════════════════════════════════════════════════
# AGENT 2: CLAUSE ANALYZER
# ═══════════════════════════════════════════════════════════════

async def analyzer_agent(state: UnderwritingState) -> UnderwritingState:
    """Detect clauses and search corpus using Insurance Server"""
    logger.info("🔍 AGENT 2: ANALYZER — Detecting clauses...")
    state["agent_logs"].append("🔍 ANALYZER: Detecting clause types via Insurance Server...")
    
    if state.get("parse_error"):
        state["agent_logs"].append("⚠️ ANALYZER: Skipped — parser failed")
        return state
    
    try:
        # Use process_submission or search_corpus to get categories
        # Since detect_categories doesn't exist, we'll use a set of common clause categories
        # or extract from the policy data
        
        policy_type = state.get("policy_data", {}).get("policy_type", "")
        
        # Define common clause categories based on policy type
        common_categories = [
            "cap on liability",
            "indemnification", 
            "warranty",
            "termination",
            "confidentiality",
            "limitation of liability",
            "insurance requirements",
            "intellectual property"
        ]
        
        # Try to get more specific categories from the insurance server
        # Use search_corpus with the policy type as query to get relevant categories
        try:
            search_results = await mcp_manager.search_corpus(policy_type if policy_type else "insurance clause", top_k=5)
            if search_results and isinstance(search_results, list):
                # Extract potential categories from search results
                extracted_categories = set()
                for result in search_results[:3]:
                    if isinstance(result, dict) and "text" in result:
                        text = result.get("text", "").lower()
                        for cat in common_categories:
                            if cat in text:
                                extracted_categories.add(cat)
                if extracted_categories:
                    categories = list(extracted_categories)
                else:
                    categories = common_categories[:5]
            else:
                categories = common_categories[:5]
        except Exception as search_e:
            logger.warning(f"Search for categories failed: {search_e}")
            categories = common_categories[:5]
        
        state["detected_categories"] = categories
        
        state["agent_logs"].append(
            f"✅ ANALYZER: Found {len(categories)} clause types: {', '.join(categories[:5])}"
        )
        
        # Search corpus for each category
        search_results = {}
        for cat in categories:
            try:
                results = await mcp_manager.search_corpus(cat, top_k=3)
                if isinstance(results, list):
                    search_results[cat] = results
                else:
                    search_results[cat] = []
                state["agent_logs"].append(
                    f"   📄 '{cat}': {len(search_results[cat])} matches found"
                )
            except Exception as e:
                state["agent_logs"].append(f"   ⚠️ '{cat}': search failed — {e}")
                search_results[cat] = []
        
        state["search_results"] = search_results
        
    except Exception as e:
        state["agent_logs"].append(f"❌ ANALYZER: Failed — {e}")
        state["errors"].append(f"Analyzer error: {e}")
        logger.error(f"Analyzer error: {e}", exc_info=True)
    
    return state
# ═══════════════════════════════════════════════════════════════
# AGENT 3: RISK ASSESSOR (Risk Server)
# ═══════════════════════════════════════════════════════════════

async def risk_assessor_agent(state: UnderwritingState) -> UnderwritingState:
    """Assess risk using Risk Server - Fixed for actual response format"""
    logger.info("🔍 AGENT 3: RISK ASSESSOR — Using Risk Server...")
    state["agent_logs"].append("📊 RISK ASSESSOR: Analyzing risk via Risk Server...")
    
    if not state.get("full_text"):
        state["agent_logs"].append("⚠️ RISK ASSESSOR: No text to analyze")
        return state
    
    try:
        policy_type = safe_get(state.get("policy_data", {}), "policy_type")
        
        # Call risk server
        risk_result = await mcp_manager.analyze_submission_risk(
            full_text=state["full_text"],
            policy_type=policy_type
        )
        
        # Store the raw result
        state["risk_assessment"] = risk_result if isinstance(risk_result, dict) else {"raw": str(risk_result)}
        
        # Initialize variables
        overall_risk = "UNKNOWN"
        avg_rating = 0.0
        clause_analyses = []
        
        # Extract data from the actual response structure
        if isinstance(risk_result, dict):
            # Look for clauses in the response
            clauses_data = risk_result.get("clauses", [])
            
            # Also check section_details for more detailed clause info
            section_details = risk_result.get("section_details", [])
            
            # Process clauses from the main "clauses" array
            if clauses_data and isinstance(clauses_data, list):
                for clause in clauses_data:
                    if isinstance(clause, dict):
                        # Extract rating (using predicted_rating as seen in debug output)
                        rating = clause.get("predicted_rating", 
                                           clause.get("rating", 
                                           clause.get("score", 0)))
                        
                        # Extract category
                        category = clause.get("category", "Unknown")
                        
                        # Extract stars
                        stars = clause.get("stars", "☆☆☆☆☆")
                        
                        # Extract preview text
                        clause_text = clause.get("clause_preview", 
                                                clause.get("text", 
                                                clause.get("clause_text", "")))
                        
                        clause_analyses.append({
                            "clause_text": str(clause_text)[:200],
                            "rating": rating,
                            "risk_level": "N/A",  # Will calculate from rating
                            "stars": stars,
                            "category": category
                        })
                        
                        state["agent_logs"].append(f"   📄 {category}: {rating}★ {stars}")
            
            # Also process section_details for additional context
            if section_details and isinstance(section_details, list):
                for section in section_details:
                    section_result = section.get("result", {})
                    rated_clauses = section_result.get("rated_clauses", [])
                    
                    for clause in rated_clauses:
                        rating = clause.get("predicted_rating", 0)
                        category = clause.get("category", "Unknown")
                        stars = clause.get("stars", "☆☆☆☆☆")
                        
                        # Check if we already have this category to avoid duplicates
                        if not any(c.get("category") == category for c in clause_analyses):
                            clause_analyses.append({
                                "clause_text": clause.get("clause_preview", "")[:200],
                                "rating": rating,
                                "risk_level": "N/A",
                                "stars": stars,
                                "category": category
                            })
            
            # Calculate overall metrics from clause ratings
            ratings = []
            for ca in clause_analyses:
                rating_val = ca.get("rating")
                if rating_val and rating_val != "N/A":
                    try:
                        ratings.append(float(rating_val))
                    except (ValueError, TypeError):
                        pass
            
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                
                # Determine overall risk from average rating
                if avg_rating >= 4.0:
                    overall_risk = "LOW RISK - Strong protections"
                elif avg_rating >= 3.0:
                    overall_risk = "MEDIUM RISK - Acceptable with conditions"
                elif avg_rating >= 2.0:
                    overall_risk = "HIGH RISK - Needs significant revision"
                else:
                    overall_risk = "CRITICAL RISK - Major concerns"
                
                state["agent_logs"].append(f"📊 Calculated from {len(ratings)} clause ratings")
            
            # Also check if risk_assessment field exists
            risk_assessment = risk_result.get("risk_assessment", {})
            if risk_assessment:
                if isinstance(risk_assessment, dict):
                    if "overall_risk" in risk_assessment:
                        overall_risk = risk_assessment.get("overall_risk", overall_risk)
                    if "average_rating" in risk_assessment:
                        avg_rating = float(risk_assessment.get("average_rating", avg_rating))
        
        # If no clauses were found, create a default analysis
        if not clause_analyses and state.get("full_text"):
            clause_analyses.append({
                "clause_text": state["full_text"][:200],
                "rating": 2.5,
                "risk_level": "MEDIUM",
                "stars": "★★★☆☆",
                "category": "General"
            })
            avg_rating = 2.5
            overall_risk = "MEDIUM RISK - Requires review"
        
        # Convert rating to float
        try:
            avg_rating = float(avg_rating)
        except (ValueError, TypeError):
            avg_rating = 0.0
        
        # Store results
        state["clause_analyses"] = clause_analyses
        state["risk_assessment"]["overall_risk"] = overall_risk
        state["risk_assessment"]["average_rating"] = avg_rating
        
        # Log summary
        state["agent_logs"].append(
            f"✅ RISK ASSESSOR: Overall Risk = {overall_risk} | Avg Rating = {avg_rating:.1f}/5.0"
        )
        state["agent_logs"].append(
            f"   Analyzed {len(clause_analyses)} clauses"
        )
        
    except Exception as e:
        state["agent_logs"].append(f"❌ Risk assessment failed: {e}")
        state["errors"].append(f"Risk assessor error: {e}")
        state["risk_assessment"] = {"error": str(e)}
        logger.error(f"Risk assessment error: {e}", exc_info=True)
    
    return state
# ═══════════════════════════════════════════════════════════════
# AGENT 4: UNDERWRITING ADVISOR
# ═══════════════════════════════════════════════════════════════

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
    # In advisor_agent, when building strong_points and negotiation_points:

    for clause in clauses:
        # Get rating (handle both 'rating' and 'predicted_rating')
        rating = clause.get("rating", clause.get("predicted_rating", 0))
        
        # Better clause description - use category and preview together
        category = clause.get("category", "Unknown")
        clause_preview = clause.get("clause_text", clause.get("clause_preview", ""))[:80]
        
        # Create a meaningful description
        if clause_preview and len(clause_preview) > 10:
            clause_description = f"{category}: {clause_preview}"
        else:
            clause_description = f"{category} clause"
        
        stars = clause.get("stars", "☆☆☆☆☆")
        
        try:
            rating_float = float(rating) if rating not in ["N/A", None, ""] else 0
        except (ValueError, TypeError):
            rating_float = 0
        
        if rating_float >= 4.0:
            strong_points.append({
                "clause": clause_description,
                "rating": rating_float,
                "stars": stars,
                "action": "Well-drafted, no changes needed"
            })
        elif rating_float >= 2.5:
            negotiation_points.append({
                "clause": clause_description,
                "rating": rating_float,
                "stars": stars,
                "action": "Consider strengthening language",
                "priority": "SHOULD"
            })
        else:
            negotiation_points.append({
                "clause": clause_description,
                "rating": rating_float,
                "stars": stars,
                "action": "Must revise this clause",
                "priority": "MUST"
            })
    
    # Sort negotiation points by priority (MUST first)
    priority_order = {"MUST": 0, "SHOULD": 1}
    negotiation_points.sort(key=lambda x: priority_order.get(x.get("priority", "SHOULD"), 1))
    
    state["strong_points"] = strong_points
    state["negotiation_points"] = negotiation_points
    
    # Make decision based on ratings
    avg_rating = safe_get(risk, "average_rating", 0) if isinstance(risk, dict) else 0
    try:
        avg_rating = float(avg_rating)
    except (ValueError, TypeError):
        avg_rating = 0.0
    
    must_count = sum(1 for n in negotiation_points if n.get("priority") == "MUST")
    
    # Decision logic
    if avg_rating >= 4.0 and must_count == 0:
        decision = "ACCEPT — Strong policy with good protections"
        emoji = "✅"
    elif avg_rating >= 3.0 and must_count <= 1:
        decision = "ACCEPT WITH CONDITIONS — Minor revisions needed"
        emoji = "⚠️"
    elif avg_rating >= 2.5 and must_count <= 3:
        decision = "REFER TO SENIOR UNDERWRITER — Significant concerns to address"
        emoji = "🔶"
    else:
        decision = "REJECT — Major risk factors present, unacceptable terms"
        emoji = "🔴"
    
    state["final_decision"] = decision
    state["decision_emoji"] = emoji
    state["agent_logs"].append(f"🎯 ADVISOR: {emoji} {decision}")
    state["agent_logs"].append(f"   Average rating: {avg_rating:.1f}/5.0, Must-fix issues: {must_count}")
    
    return state



# ═══════════════════════════════════════════════════════════════
# AGENT 5: REPORTER
# ═══════════════════════════════════════════════════════════════
async def reporter_agent(state: UnderwritingState) -> UnderwritingState:
    """Generate final report with improved formatting"""
    logger.info("🔍 AGENT 5: REPORTER — Generating report...")
    state["agent_logs"].append("📋 REPORTER: Generating final report...")
    
    policy = state.get("policy_data", {})
    risk = state.get("risk_assessment", {})
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["report_generated_at"] = now
    
    # Get rating and risk
    avg_rating = safe_get(risk, "average_rating", 0)
    overall_risk = safe_get(risk, "overall_risk", "N/A")
    
    # Format rating with stars
    rating_stars = "⭐" * min(5, int(avg_rating)) if avg_rating else "☆☆☆☆☆"
    
    # Create executive summary
    lines = [
        "=" * 60,
        "     MULTI-AGENT UNDERWRITING REPORT",
        "=" * 60,
        "",
        f"📄 POLICY: {safe_get(policy, 'policy_number', 'N/A')}",
        f"🏢 INSURED: {safe_get(policy, 'insured_name', 'N/A')}",
        f"📋 TYPE: {safe_get(policy, 'policy_type', 'N/A')}",
        f"📅 PERIOD: {safe_get(policy, 'effective_date', 'N/A')} → {safe_get(policy, 'expiration_date', 'N/A')}",
        "",
        f"📊 OVERALL RISK: {overall_risk}",
        f"⭐ AVERAGE RATING: {avg_rating}/5.0 {rating_stars}",
        "",
        f"🎯 DECISION: {state.get('decision_emoji', '')} {state.get('final_decision', 'N/A')}",
    ]
    
    # Strong points - show actual clause details
    strong_points = state.get("strong_points", [])
    if strong_points:
        lines.append("")
        lines.append("🟢 STRONG POINTS (Well-rated clauses):")
        for idx, sp in enumerate(strong_points[:5], 1):
            clause_text = sp.get('clause', '')[:80]
            rating = sp.get('rating', 0)
            stars = sp.get('stars', '')
            lines.append(f"   {idx}. {stars} {rating}★ - {clause_text}")
    
    # Required actions - show actual revision needs
    negotiation_points = state.get("negotiation_points", [])
    if negotiation_points:
        lines.append("")
        lines.append("🔴 REQUIRED ACTIONS (Clauses needing attention):")
        for idx, np in enumerate(negotiation_points[:8], 1):
            priority = np.get("priority", "SHOULD")
            priority_emoji = "🔴" if priority == "MUST" else "🟡"
            clause_text = np.get('clause', '')[:70]
            rating = np.get('rating', 0)
            action = np.get('action', 'Review required')
            lines.append(f"   {idx}. {priority_emoji} [{priority}] {action}")
            lines.append(f"      📝 {clause_text} (Rating: {rating}★)")
    
    # Summary statistics
    lines.extend([
        "",
        "📊 SUMMARY STATISTICS:",
        f"   • Total clauses analyzed: {len(state.get('clause_analyses', []))}",
        f"   • Strong points: {len(strong_points)}",
        f"   • Required actions: {len(negotiation_points)}",
        f"   • MUST fix issues: {sum(1 for n in negotiation_points if n.get('priority') == 'MUST')}",
    ])
    
    # Add timestamp
    lines.extend([
        "",
        f"📅 Report generated: {now}",
        "",
        "🤖 Pipeline: Insurance Server → Risk Server → Advisor → Report",
    ])
    
    state["executive_summary"] = "\n".join(lines)
    
    # Create full report with more details
    full_sections = [
        state["executive_summary"],
        "",
        "=" * 60,
        "DETAILED CLAUSE ANALYSIS",
        "=" * 60,
    ]
    
    # Add all clause analyses
    for idx, clause in enumerate(state.get("clause_analyses", []), 1):
        category = clause.get("category", "Unknown")
        rating = clause.get("rating", "N/A")
        stars = clause.get("stars", "☆☆☆☆☆")
        clause_text = clause.get("clause_text", "")[:150]
        
        full_sections.extend([
            f"\n{idx}. {category.upper()}",
            f"   Rating: {rating}/5.0 {stars}",
            f"   Text: {clause_text}...",
        ])
    
    # Add agent logs
    full_sections.extend([
        "",
        "=" * 60,
        "AGENT PROCESSING LOGS",
        "=" * 60,
    ])
    full_sections.extend(state["agent_logs"])
    
    state["full_report"] = "\n".join(full_sections)
    
    state["agent_logs"].append("✅ REPORTER: Report generated successfully")
    return state
# ═══════════════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════

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
    
    logger.info(f"🚀 Starting pipeline for submission: {submission_id}")
    result = await app.ainvoke(initial_state, config)
    logger.info(f"✅ Pipeline complete for submission: {submission_id}")
    
    return result


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    async def main():
        import sys
        if len(sys.argv) < 2:
            print("Usage: python underwriting_graph.py <pdf_path>")
            sys.exit(1)
        
        pdf_path = sys.argv[1]
        result = await run_underwriting_pipeline(pdf_path)
        print(result["executive_summary"])
    
    asyncio.run(main())