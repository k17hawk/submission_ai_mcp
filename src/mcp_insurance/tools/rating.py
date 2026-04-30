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
        logger.info(f"Rating clause for category: {category or 'global'}")
        rating = _ratings_model.predict_rating(clause_text, category=category)
        logger.info(f"Predicted rating: {rating:.2f} for category: {category or 'global'}")
        return {
            "clause_text": clause_text[:200] + "..." if len(clause_text) > 200 else clause_text,
            "predicted_rating": rating,
            "category_used": category or "global",
            "error": None
        }
    except Exception as e:
        logger.error(f"Error occurred while rating clause: {e}")
        return {"error": str(e), "predicted_rating": None}

def get_available_categories() -> List[str]:
    """
    Return all rating category names from the ACORD Excel model.
    Useful for mapping query strings to exact category names.
    """
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
        category: Category name (e.g., "cap on liability").
        top_k: Number of examples to return.
        sample: How to choose examples:
            "first"  – first rows from the spreadsheet (original behaviour).
            "best"   – clauses with the highest ratings.
            "random" – a random sample (for diversity).
        query_text: If provided, return the clauses most similar to this text
                    (using TF-IDF cosine similarity) instead of the above methods.
        deduplicate: Remove duplicate clause texts within the category (keeps the highest rating).
        max_text_len: Max characters before truncation. If None or 0, full text is returned.

    Returns:
        Dictionary with category and list of examples (clause_text, rating, stars, ...).
    """
    _ensure_ratings_model()
    df = _ratings_model.get_clauses_by_category(category)
    if df.empty:
        return {"error": f"Category '{category}' not found", "examples": []}

    # Optional deduplication (keep best rating for each unique text)
    if deduplicate:
        df = df.sort_values('rating', ascending=False).drop_duplicates(
            subset='clause_text', keep='first'
        )

    # Select rows based on mode
    if query_text:
        logger.info(f"Finding {top_k} examples most similar to query for category: {category}")
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

    # Build output list
    examples = []
    for _, row in selected.iterrows():
        full_text = row['clause_text']
        rating = row['rating']

        # Truncate only if max_text_len is a positive number
        if max_text_len and max_text_len > 0 and len(full_text) > max_text_len:
            display_text = full_text[:max_text_len] + "…"
        else:
            display_text = full_text

        # Round to nearest 0.5 star for display
        star_count = round(rating * 2) / 2
        stars = ''.join(['★' if i < int(star_count) else '½' if i < star_count and star_count % 1 else '☆'
                         for i in range(5)])

        examples.append({
            "clause_text": display_text,
            "rating": rating,
            "stars": stars,
        })

    logger.info(f"Returning {len(examples)} examples for category: {category}")
    return {"category": category, "examples": examples}