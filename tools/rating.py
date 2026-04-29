# tools/rating.py
"""
MCP tools for rating insurance clauses using the ACORD clause pairs Excel.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from core.clause_ratings import ClauseRatings

# Global model cache
_ratings_model: Optional[ClauseRatings] = None


def _ensure_ratings_model():
    global _ratings_model
    if _ratings_model is None:
        from dataset_paths import DatasetPaths
        paths = DatasetPaths()
        _ratings_model = ClauseRatings(paths.rating_excel)
        _ratings_model.build_prediction_model()


async def rate_clause(clause_text: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    MCP Tool: Predict a rating (1-5 stars) for a given insurance clause.

    Args:
        clause_text: The clause text to rate.
        category: Optional category (e.g., "cap on liability") to constrain prediction.

    Returns:
        Dictionary with predicted_rating (float) and confidence (optional).
    """
    _ensure_ratings_model()
    try:
        rating = _ratings_model.predict_rating(clause_text, category=category)
        return {
            "clause_text": clause_text[:200] + "..." if len(clause_text) > 200 else clause_text,
            "predicted_rating": rating,
            "category_used": category or "global",
            "error": None
        }
    except Exception as e:
        return {"error": str(e), "predicted_rating": None}


async def get_rating_examples(category: str, top_k: int = 5) -> Dict[str, Any]:
    """
    MCP Tool: Retrieve example clauses and their ratings for a given category.

    Args:
        category: One of the categories from the Excel (e.g., "cap on liability").
        top_k: Number of examples to return.

    Returns:
        Dictionary with category and list of (clause_preview, rating).
    """
    _ensure_ratings_model()
    df = _ratings_model.get_clauses_by_category(category)
    if df.empty:
        return {"error": f"Category '{category}' not found", "examples": []}
    examples = df.head(top_k)[['clause_text', 'rating']].to_dict(orient='records')
    # Truncate long clause texts
    for ex in examples:
        if len(ex['clause_text']) > 300:
            ex['clause_text'] = ex['clause_text'][:300] + "..."
    return {"category": category, "examples": examples}