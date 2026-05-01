# tools/risk.py
"""
Risk assessment tool for submission pipeline output.
Uses the star ratings to determine risk levels and flags missing protections.
"""

from typing import Dict, Any, List, Optional

# ─── Risk thresholds ───
RISK_THRESHOLDS = {
    "LOW": 4.0,      # ≥ 4.0 stars → low risk
    "MEDIUM": 2.5,   # ≥ 2.5 stars → medium risk
    "HIGH": 0.0,     # < 2.5 stars → high risk
}

# ─── Expected clauses per policy type ───
EXPECTED_CLAUSES = {
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
        "indemnification carveout to cap on liability",
        "fraud, negligence or willful misconduct carveout to liability cap",
    ]
}

# ─── Clause descriptions (for human-readable reports) ───
CLAUSE_DESCRIPTIONS = {
    "cap on liability": "Limits total liability to a fixed amount or multiple of fees",
    "indemnification carveout to cap on liability": "Indemnification obligations are not subject to the liability cap",
    "IP infringement exception to cap on liability": "IP infringement claims are not subject to the liability cap",
    "fraud, negligence or willful misconduct carveout to liability cap": "Fraud or willful misconduct is not subject to the liability cap",
    "confidentiality exceptions to liability cap": "Confidentiality breaches are not subject to the liability cap",
    "personal or bodily injury exception to liability cap": "Personal/bodily injury claims are not subject to the liability cap",
    "compliance with law carveout to cap on liability": "Legal/regulatory violations are not subject to the liability cap",
    "mutual liability cap": "Liability cap applies equally to both parties",
    "third party IP infringement exception to cap on liability": "Third-party IP claims carved out via indemnification",
}


def _get_risk_level(rating: float) -> str:
    """Convert rating to risk level."""
    if rating >= RISK_THRESHOLDS["LOW"]:
        return "LOW"
    elif rating >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "HIGH"


def _get_risk_emoji(risk_level: str) -> str:
    """Return emoji for risk level."""
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk_level, "⚪")


def _get_risk_summary(rating: float) -> str:
    """Human-readable risk interpretation."""
    if rating >= 4.5:
        return "Very buyer‑friendly — strongly recommend accepting"
    elif rating >= 3.5:
        return "Buyer‑friendly — acceptable terms"
    elif rating >= 2.5:
        return "Neutral — standard market terms, no major concerns"
    elif rating >= 1.5:
        return "Seller‑friendly — review carefully before accepting"
    else:
        return "Very seller‑friendly — significant risk, consider renegotiation"


async def assess_submission_risk(
    submission_report: Dict[str, Any],
    policy_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    MCP Tool: Assess overall risk of a submission based on clause ratings.

    Args:
        submission_report: The output from process_submission.
        policy_type: Optional policy type to check for missing expected clauses.

    Returns:
        Dictionary with overall_risk, average_rating, clause_risks, and alerts.
    """
    if submission_report.get("error"):
        return {"error": "Cannot assess risk — submission has errors", "overall_risk": "UNKNOWN"}

    results = submission_report.get("results", [])
    if not results:
        return {"error": "No clauses analyzed", "overall_risk": "UNKNOWN"}

    clause_risks = []
    total_rating = 0.0
    count = 0
    detected_categories = []

    for query_group in results:
        query = query_group["query"]
        detected_categories.append(query)
        clauses = query_group.get("rated_clauses", [])

        if not clauses:
            clause_risks.append({
                "category": query,
                "description": CLAUSE_DESCRIPTIONS.get(query, ""),
                "risk_level": "HIGH",
                "emoji": "🔴",
                "best_rating": None,
                "stars": "☆☆☆☆☆",
                "reason": "No matching clauses found in corpus",
                "clauses_found": 0,
            })
            continue

        best = clauses[0]
        best_rating = best.get("predicted_rating")
        
        if best_rating is not None:
            total_rating += best_rating
            count += 1
            risk_level = _get_risk_level(best_rating)
        else:
            risk_level = "HIGH"

        clause_risks.append({
            "category": query,
            "description": CLAUSE_DESCRIPTIONS.get(query, ""),
            "risk_level": risk_level,
            "emoji": _get_risk_emoji(risk_level),
            "best_rating": best_rating,
            "stars": best.get("stars", "☆☆☆☆☆"),
            "reason": _get_risk_summary(best_rating) if best_rating else "Could not rate",
            "best_clause_id": best.get("doc_id"),
            "best_clause_score": best.get("score"),
            "clauses_found": len(clauses),
        })

    # Sort: HIGH risk first, then MEDIUM, then LOW
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    clause_risks.sort(key=lambda x: risk_order.get(x["risk_level"], 3))

    # ─── Overall score ───
    avg_rating = round(total_rating / count, 1) if count > 0 else 0.0
    overall_risk = _get_risk_level(avg_rating) if count > 0 else "UNKNOWN"

    # Count by risk level
    high_count = sum(1 for c in clause_risks if c["risk_level"] == "HIGH")
    medium_count = sum(1 for c in clause_risks if c["risk_level"] == "MEDIUM")
    low_count = sum(1 for c in clause_risks if c["risk_level"] == "LOW")

    # ─── Missing expected clauses ───
    missing = []
    if policy_type:
        expected = EXPECTED_CLAUSES.get(policy_type, EXPECTED_CLAUSES["default"])
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
                    "reason": "Expected protection not detected in submission",
                })

    # ─── Alerts ───
    alerts = []
    if high_count > 0:
        alerts.append(f"🔴 {high_count} clause(s) have HIGH risk — review required")
    if missing:
        alerts.append(f"⚠️ {len(missing)} expected protection(s) missing from policy")
    if avg_rating < 3.0:
        alerts.append("🔴 Overall rating below 3.0 — consider renegotiation")
    elif avg_rating < 3.5:
        alerts.append("🟡 Overall rating below 3.5 — review recommended")

    return {
        "overall_risk": overall_risk,
        "overall_risk_emoji": _get_risk_emoji(overall_risk),
        "average_rating": avg_rating,
        "average_stars": _make_stars(avg_rating),
        "total_clauses_analyzed": count,
        "high_risk_count": high_count,
        "medium_risk_count": medium_count,
        "low_risk_count": low_count,
        "clause_risks": clause_risks,
        "missing_protections": missing,
        "alerts": alerts,
        "recommendation": _get_risk_summary(avg_rating),
    }


def _make_stars(rating: float) -> str:
    """Convert numeric rating to 5-star display string."""
    star_count = round(rating * 2) / 2
    return ''.join([
        '★' if i < int(star_count)
        else '½' if i < star_count and star_count % 1
        else '☆'
        for i in range(5)
    ])