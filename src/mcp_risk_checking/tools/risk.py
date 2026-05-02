"""
Risk assessment tools for the Risk Checking MCP Server.
Analyzes a set of rated clauses and produces a risk report.
"""

from typing import Dict, Any, List, Optional

# ─── Thresholds ───
RISK_THRESHOLDS = {"LOW": 4.0, "MEDIUM": 2.5, "HIGH": 0.0}

# ─── Expected protections per policy type ───
EXPECTED_PROTECTIONS = {
    "Commercial General Liability": [
        "cap on liability",
        "indemnification carveout to cap on liability",
        "IP infringement exception to cap on liability",
        "fraud, negligence or willful misconduct carveout to liability cap",
        "confidentiality exceptions to liability cap",
        "personal or bodily injury exception to liability cap",
        "compliance with law carveout to cap on liability",
    ],
    "default": [
        "cap on liability",
        "fraud, negligence or willful misconduct carveout to liability cap",
    ]
}

CLAUSE_DESCRIPTIONS = {
    "cap on liability": "Limits total liability to a fixed amount or fee multiple",
    "indemnification carveout to cap on liability": "Indemnity obligations exceed the cap",
    "IP infringement exception to cap on liability": "IP claims not limited by the cap",
    "fraud, negligence or willful misconduct carveout to liability cap": "Fraud/misconduct not limited by cap",
    "confidentiality exceptions to liability cap": "Confidentiality breaches not limited by cap",
    "personal or bodily injury exception to liability cap": "Injury/death claims not limited by cap",
    "compliance with law carveout to cap on liability": "Regulatory violations not limited by cap",
}


def _risk_level(rating: float) -> str:
    if rating >= RISK_THRESHOLDS["LOW"]:
        return "LOW"
    elif rating >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"


def _emoji(level: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "MISSING": "⚠️"}.get(level, "⚪")


def _interpretation(rating: float) -> str:
    if rating >= 4.5:
        return "Very buyer‑friendly — strongly recommend"
    elif rating >= 3.5:
        return "Buyer‑friendly — acceptable terms"
    elif rating >= 2.5:
        return "Neutral — standard market terms"
    elif rating >= 1.5:
        return "Seller‑friendly — review carefully"
    return "Very seller‑friendly — significant risk"


async def assess_risk(
    clauses: List[Dict[str, Any]],
    policy_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assess risk for a list of rated clauses.

    Args:
        clauses: List of dicts with keys: category, predicted_rating, stars
        policy_type: Optional policy type to check for missing protections.

    Returns:
        Risk report with overall_risk, per-clause breakdown, missing protections, alerts.
    """
    if not clauses:
        return {"error": "No clauses provided", "overall_risk": "UNKNOWN"}

    clause_risks = []
    total = 0.0
    count = 0
    detected_categories = []

    for clause in clauses:
        category = clause.get("category", "unknown")
        rating = clause.get("predicted_rating")
        stars = clause.get("stars", "☆☆☆☆☆")
        detected_categories.append(category)

        if rating is None:
            clause_risks.append({
                "category": category,
                "description": CLAUSE_DESCRIPTIONS.get(category, ""),
                "risk_level": "HIGH",
                "emoji": "🔴",
                "rating": None,
                "stars": "☆☆☆☆☆",
                "interpretation": "Could not rate this clause",
            })
            continue

        total += rating
        count += 1
        level = _risk_level(rating)

        clause_risks.append({
            "category": category,
            "description": CLAUSE_DESCRIPTIONS.get(category, ""),
            "risk_level": level,
            "emoji": _emoji(level),
            "rating": round(rating, 1),
            "stars": stars,
            "interpretation": _interpretation(rating),
        })

    # Sort: HIGH first
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    clause_risks.sort(key=lambda x: risk_order.get(x["risk_level"], 3))

    # Overall
    avg_rating = round(total / count, 1) if count > 0 else 0.0
    overall_risk = _risk_level(avg_rating) if count > 0 else "UNKNOWN"

    high_count = sum(1 for c in clause_risks if c["risk_level"] == "HIGH")
    medium_count = sum(1 for c in clause_risks if c["risk_level"] == "MEDIUM")
    low_count = sum(1 for c in clause_risks if c["risk_level"] == "LOW")

    # Missing protections
    missing = []
    if policy_type:
        expected = EXPECTED_PROTECTIONS.get(policy_type, EXPECTED_PROTECTIONS["default"])
        for expected_cat in expected:
            found = any(
                expected_cat.lower() in cat.lower() or cat.lower() in expected_cat.lower()
                for cat in detected_categories
            )
            if not found:
                missing.append({
                    "category": expected_cat,
                    "description": CLAUSE_DESCRIPTIONS.get(expected_cat, ""),
                    "risk": "MISSING",
                    "emoji": "⚠️",
                    "reason": "Expected protection not detected",
                })

    # Alerts
    alerts = []
    if high_count > 0:
        alerts.append(f"🔴 {high_count} high-risk clause(s) — review required")
    if missing:
        alerts.append(f"⚠️ {len(missing)} expected protection(s) missing")
    if avg_rating < 3.0:
        alerts.append("🔴 Overall rating below 3.0 — consider rejection")
    elif avg_rating < 3.5:
        alerts.append("🟡 Overall rating below 3.5 — review recommended")

    return {
        "overall_risk": overall_risk,
        "overall_risk_emoji": _emoji(overall_risk),
        "average_rating": avg_rating,
        "average_stars": _make_stars(avg_rating) if count > 0 else "☆☆☆☆☆",
        "clauses_analyzed": count,
        "high_risk_count": high_count,
        "medium_risk_count": medium_count,
        "low_risk_count": low_count,
        "clause_risks": clause_risks,
        "missing_protections": missing,
        "alerts": alerts,
        "recommendation": _interpretation(avg_rating) if count > 0 else "Unable to assess",
    }


def _make_stars(rating: float) -> str:
    star_count = round(rating * 2) / 2
    return ''.join([
        '★' if i < int(star_count)
        else '½' if i < star_count and star_count % 1
        else '☆'
        for i in range(5)
    ])