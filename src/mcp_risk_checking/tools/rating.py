# servers/risk_checker/tools/rating.py
"""
Rating tools for the Risk Checking MCP Server.
Uses the ACORD clause pairs Excel to rate insurance clauses.
"""

from typing import Dict, Any, List, Optional, Literal
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.mcp_insurance.core.clause_ratings import ClauseRatings
from src.mcp_insurance.data.dataset_paths import DatasetPaths

_ratings_model: Optional[ClauseRatings] = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_model():
    global _ratings_model
    if _ratings_model is None:
        paths = DatasetPaths()
        _ratings_model = ClauseRatings(paths.rating_excel)
        _ratings_model.build_prediction_model()
        logger.info("✅ Risk Checker: Rating model loaded")


def _make_stars(rating: float) -> str:
    """Convert numeric rating to 5-star string."""
    star_count = round(rating * 2) / 2
    return ''.join([
        '★' if i < int(star_count)
        else '½' if i < star_count and star_count % 1
        else '☆'
        for i in range(5)
    ])


async def rate_clause(clause_text: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Rate a single insurance clause on a 1.0–5.0 scale.

    Args:
        clause_text: The full text of the clause to rate.
        category: Optional category name (e.g., "cap on liability") for scoped rating.

    Returns:
        Dictionary with predicted_rating, stars, category_used, and clause_preview.
    """
    _ensure_model()
    try:
        rating = _ratings_model.predict_rating(clause_text, category=category)
        stars = _make_stars(rating)
        preview = clause_text[:200] + "..." if len(clause_text) > 200 else clause_text

        logger.info(f"Rated clause [{category or 'global'}]: {rating:.1f} → {stars}")

        return {
            "clause_preview": preview,
            "predicted_rating": round(rating, 1),
            "stars": stars,
            "category_used": category or "global",
            "error": None
        }
    except Exception as e:
        logger.error(f"Rating failed: {e}")
        return {
            "clause_preview": clause_text[:200],
            "predicted_rating": None,
            "stars": "☆☆☆☆☆",
            "category_used": category or "global",
            "error": str(e)
        }


async def get_rating_examples(
    category: str,
    top_k: int = 5,
    sample: Literal["first", "best", "random"] = "first",
    query_text: Optional[str] = None,
    deduplicate: bool = True,
    max_text_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get example clauses and their ratings for a given category.

    Args:
        category: Category name (case-insensitive, e.g., "cap on liability").
        top_k: Number of examples to return.
        sample: "first", "best", or "random".
        query_text: If provided, return clauses most similar to this text.
        deduplicate: Remove duplicate clause texts.
        max_text_len: Max characters per example before truncation.

    Returns:
        Dictionary with category name and examples list.
    """
    _ensure_model()

    # Case-insensitive category lookup
    all_categories = _ratings_model.list_categories()
    exact_category = None
    for cat in all_categories:
        if cat.strip().lower() == category.strip().lower():
            exact_category = cat
            break

    if exact_category is None:
        return {
            "error": f"Category '{category}' not found. Available: {all_categories[:10]}...",
            "examples": []
        }

    df = _ratings_model.get_clauses_by_category(exact_category)
    if df.empty:
        return {"error": f"No clauses for '{exact_category}'", "examples": []}

    if deduplicate:
        df = df.sort_values('rating', ascending=False).drop_duplicates(
            subset='clause_text', keep='first'
        )

    # Select rows
    if query_text:
        try:
            vec = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vec.fit_transform(df['clause_text'])
            query_vec = vec.transform([query_text])
            sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            selected = df.iloc[top_indices]
        except Exception as e:
            return {"error": f"Similarity search failed: {e}", "examples": []}
    elif sample == "best":
        selected = df.sort_values('rating', ascending=False).head(top_k)
    elif sample == "random":
        selected = df.sample(n=min(top_k, len(df)), random_state=42)
    else:
        selected = df.head(top_k)

    examples = []
    for _, row in selected.iterrows():
        full_text = row['clause_text']
        rating_val = row['rating']

        if max_text_len and len(full_text) > max_text_len:
            display = full_text[:max_text_len] + "…"
        else:
            display = full_text

        examples.append({
            "clause_text": display,
            "rating": round(rating_val, 1),
            "stars": _make_stars(rating_val),
        })

    logger.info(f"Returned {len(examples)} examples for '{exact_category}'")
    return {"category": exact_category, "examples": examples}


async def get_available_categories() -> List[str]:
    """Return all available rating categories from the Excel file."""
    _ensure_model()
    return _ratings_model.list_categories()