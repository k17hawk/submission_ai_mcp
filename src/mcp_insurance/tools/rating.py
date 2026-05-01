# tools/rating.py
"""
MCP tools for rating insurance clauses using the ACORD clause pairs Excel.
"""

from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor

from src.mcp_insurance.core.clause_ratings import ClauseRatings
from src.mcp_insurance.data.dataset_paths import DatasetPaths

# Global model cache
_ratings_model: Optional[ClauseRatings] = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_ratings_model():
    global _ratings_model
    if _ratings_model is None:
        paths = DatasetPaths()
        _ratings_model = ClauseRatings(paths.rating_excel)
        _ratings_model.build_prediction_model()


def _make_stars(rating: float) -> str:
    """Convert numeric rating to 5-star display string."""
    star_count = round(rating * 2) / 2
    return ''.join([
        '★' if i < int(star_count) 
        else '½' if i < star_count and star_count % 1 
        else '☆'
        for i in range(5)
    ])


async def rate_clause(clause_text: str, category: Optional[str] = None) -> Dict[str, Any]:
    """Rate a single clause text, optionally scoped to a category."""
    _ensure_ratings_model()
    try:
        logger.info(f"Rating clause for category: {category or 'global'}")
        rating = _ratings_model.predict_rating(clause_text, category=category)
        stars = _make_stars(rating)
        logger.info(f"Predicted rating: {rating:.2f} → {stars} for category: {category or 'global'}")

        return {
            "clause_text": clause_text[:200] + "..." if len(clause_text) > 200 else clause_text,
            "predicted_rating": rating,
            "stars": stars,
            "category_used": category or "global",
            "error": None
        }
    except Exception as e:
        logger.error(f"Error occurred while rating clause: {e}")
        return {"error": str(e), "predicted_rating": None, "stars": "☆☆☆☆☆", "category_used": category or "global"}


def get_available_categories() -> List[str]:
    """Return all available rating categories from the Excel file."""
    _ensure_ratings_model()
    return _ratings_model.list_categories()


async def get_rating_examples(
    category: str,
    top_k: int = 5,
    sample: Literal["first", "best", "random"] = "first",
    query_text: Optional[str] = None,
    deduplicate: bool = True,
    max_text_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    MCP Tool: Retrieve example clauses and their ratings for a given category.

    Args:
        category: Category name (e.g., "cap on liability" – case‑insensitive).
        top_k: Number of examples to return.
        sample: "first", "best", or "random".
        query_text: If provided, return clauses most similar to this text.
        deduplicate: Remove duplicate clause texts (keep highest rating).
        max_text_len: Max characters before truncation.

    Returns:
        Dictionary with "category" and "examples" list.
    """
    _ensure_ratings_model()

    # --- Case‑insensitive lookup for the exact category name ---
    all_categories = _ratings_model.list_categories()
    exact_category = None
    for cat in all_categories:
        if cat.strip().lower() == category.strip().lower():
            exact_category = cat
            break

    if exact_category is None:
        return {"error": f"Category '{category}' not found. Available: {all_categories}", "examples": []}

    df = _ratings_model.get_clauses_by_category(exact_category)
    if df.empty:
        return {"error": f"No clauses found for category '{category}'", "examples": []}

    # Optional deduplication
    if deduplicate:
        df = df.sort_values('rating', ascending=False).drop_duplicates(
            subset='clause_text', keep='first'
        )

    # Select rows
    if query_text:
        logger.info(f"Finding {top_k} examples similar to query for category '{exact_category}'")
        try:
            vec = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vec.fit_transform(df['clause_text'])
            query_vec = vec.transform([query_text])
            sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            selected = df.iloc[top_indices]
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return {"error": f"Similarity search failed: {str(e)}", "examples": []}
    elif sample == "best":
        selected = df.sort_values('rating', ascending=False).head(top_k)
    elif sample == "random":
        selected = df.sample(n=min(top_k, len(df)), random_state=42)
    else:  # "first"
        selected = df.head(top_k)

    examples = []
    for _, row in selected.iterrows():
        full_text = row['clause_text']
        rating = row['rating']

        if max_text_len and max_text_len > 0 and len(full_text) > max_text_len:
            display_text = full_text[:max_text_len] + "…"
        else:
            display_text = full_text

        stars = _make_stars(rating)

        examples.append({
            "clause_text": display_text,
            "rating": rating,
            "stars": stars,
        })

    logger.info(f"Returning {len(examples)} examples for category: {exact_category}")
    return {
        "category": exact_category,
        "examples": examples,
    }