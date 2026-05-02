"""
Single-call pipeline for the Risk Checking MCP Server.
Takes clause texts directly → rates them → returns risk assessment.
"""

import logging
from typing import Dict, Any, List, Optional


from src.mcp_risk_checking.tools.rating import rate_clause, get_available_categories
from src.mcp_risk_checking.tools.risk import assess_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Category detection keywords (same logic as submission parser) ───
DETECTION_RULES = {
    "cap on liability": [
        "limitation of liability", "liability cap", "limitation on damages",
        "liability limit", "aggregate liability", "maximum liability",
        "shall not exceed", "limited to the amount"
    ],
    "indemnification carveout to cap on liability": [
        "indemnif", "hold harmless", "indemnity"
    ],
    "IP infringement exception to cap on liability": [
        "intellectual property", "infringement", "patent", "copyright",
        "trademark", "trade secret", "unauthorized use of intellectual property"
    ],
    "fraud, negligence or willful misconduct carveout to liability cap": [
        "fraud", "gross negligence", "willful misconduct", "intentional misconduct",
        "intentional breach"
    ],
    "confidentiality exceptions to liability cap": [
        "confidentiality", "confidential information"
    ],
    "personal or bodily injury exception to liability cap": [
        "personal injury", "bodily injury", "death", "damage to tangible property"
    ],
    "compliance with law carveout to cap on liability": [
        "mandatory applicable law", "non-compliance with any mandatory",
        "compliance with law"
    ],
}


def _detect_categories(text: str) -> List[str]:
    """Auto-detect which clause categories are present in a text."""
    text_lower = text.lower()
    detected = []
    for category, keywords in DETECTION_RULES.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(category)
                break

    # Deduplicate preserving order
    seen = set()
    unique = []
    for cat in detected:
        if cat not in seen:
            seen.add(cat)
            unique.append(cat)
    return unique


async def analyze_clause_risk(
    clause_text: str,
    policy_type: Optional[str] = None,
    auto_detect_categories: bool = True,
    custom_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Single-call pipeline: take clause text → detect categories → rate → assess risk.

    Args:
        clause_text: The full text of a clause or policy section.
        policy_type: Optional policy type for missing protection checks.
        auto_detect_categories: If True, detect relevant categories from text.
        custom_categories: If provided, use these categories instead of auto-detection.

    Returns:
        Complete risk report with ratings and assessment.
    """
    logger.info("🔍 Risk pipeline: analyzing clause text")

    # ─── Step 1: Determine categories ───
    if custom_categories:
        categories = custom_categories
        logger.info(f"Using provided categories: {categories}")
    elif auto_detect_categories:
        categories = _detect_categories(clause_text)
        logger.info(f"Auto-detected {len(categories)} categories: {categories}")
    else:
        return {"error": "No categories provided and auto_detect is disabled"}

    if not categories:
        return {
            "error": "No clause categories detected",
            "clauses": [],
            "risk_assessment": None,
        }

    # ─── Step 2: Rate for each detected category ───
    rated_clauses = []
    for category in categories:
        try:
            result = await rate_clause(clause_text, category=category)
            rated_clauses.append({
                "category": category,
                "predicted_rating": result.get("predicted_rating"),
                "stars": result.get("stars", "☆☆☆☆☆"),
                "clause_preview": result.get("clause_preview"),
            })
        except Exception as e:
            logger.error(f"Failed to rate '{category}': {e}")
            rated_clauses.append({
                "category": category,
                "predicted_rating": None,
                "stars": "☆☆☆☆☆",
                "error": str(e),
            })

    # ─── Step 3: Assess overall risk ───
    risk_assessment = await assess_risk(rated_clauses, policy_type=policy_type)

    # ─── Step 4: Build complete response ───
    return {
        "clauses_rated": len(rated_clauses),
        "categories_detected": categories,
        "rated_clauses": rated_clauses,
        "risk_assessment": risk_assessment,
    }


async def analyze_submission_text(
    full_text: str,
    policy_type: Optional[str] = None,
    clause_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze a full submission text with multiple clauses.

    Args:
        full_text: The complete submission/policy text.
        policy_type: Optional policy type for missing protection checks.
        clause_texts: Optional pre-extracted clause texts. If None,
                      treats the full_text as a single clause.

    Returns:
        Risk report for all clauses found.
    """
    logger.info("📄 Risk pipeline: analyzing full submission")

    texts_to_analyze = clause_texts if clause_texts else [full_text]

    all_results = []
    for i, text in enumerate(texts_to_analyze):
        result = await analyze_clause_risk(
            clause_text=text,
            policy_type=policy_type,
            auto_detect_categories=True,
        )
        all_results.append({
            "section_index": i,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "result": result,
        })

    # ─── Aggregate across all sections ───
    all_rated = []
    for section in all_results:
        all_rated.extend(section["result"].get("rated_clauses", []))

    # Deduplicate by category (keep highest rating)
    best_per_category = {}
    for clause in all_rated:
        cat = clause["category"]
        if cat not in best_per_category or (
            clause.get("predicted_rating") or 0
        ) > (best_per_category[cat].get("predicted_rating") or 0):
            best_per_category[cat] = clause

    unique_clauses = list(best_per_category.values())

    # Final risk assessment
    risk_assessment = await assess_risk(unique_clauses, policy_type=policy_type)

    return {
        "sections_analyzed": len(texts_to_analyze),
        "unique_categories_found": len(unique_clauses),
        "clauses": unique_clauses,
        "section_details": all_results,
        "risk_assessment": risk_assessment,
    }