"""
MCP tool that orchestrates the full ACORD submission parsing pipeline:
parse PDF → search clause library → rate retrieved clauses → compile report.
"""

import logging
from typing import Dict, Any, List, Optional

from src.mcp_insurance.tools.parsing import parse_acord_submission
from src.mcp_insurance.tools.retrieval import search_corpus, get_document_by_id, _ensure_initialized as ensure_retrieval
from src.mcp_insurance.tools.rating import rate_clause, _ensure_ratings_model, get_available_categories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_submission(
    pdf_path: str,
    queries: List[str],
    top_k: int = 5,
    retrieval_method: str = "bm25",
    map_query_to_rating_category: bool = True,
) -> Dict[str, Any]:
    """
    MCP Tool: Full pipeline – parse ACORD PDF, retrieve relevant clauses,
    rate them, and generate a submission report.

    Args:
        pdf_path: Path to the ACORD form PDF.
        queries: List of clause-type queries to search for
                 (e.g., ["Cap on Liability", "Governing Law"]).
        top_k: Number of clauses to retrieve per query.
        retrieval_method: "bm25", "embedding", or "hybrid".
        map_query_to_rating_category: If True, try to match each query
            to a rating category from the ACORD Excel for better predictions.

    Returns:
        Dictionary with:
            - policy_data: parsed policy fields
            - results: list of query results, each with retrieved clauses and ratings
            - error: optional error message if PDF parsing failed
    """
    logger.info(f"🚀 Starting submission pipeline for PDF: {pdf_path}")
    report = {
        "policy_data": None,
        "results": [],
        "error": None,
    }

    # ---- Step 1: Parse PDF ----
    parse_result = await parse_acord_submission(pdf_path)
    if parse_result.get("error"):
        report["error"] = f"PDF parsing failed: {parse_result['error']}"
        return report
    report["policy_data"] = parse_result["policy_data"]

    # ---- Prepare rating category lookup if enabled ----
    category_map = {}
    if map_query_to_rating_category:
        try:
            available_categories = get_available_categories()
            # Build a lowercase -> exact mapping
            cat_lower = {c.lower(): c for c in available_categories}
            for q in queries:
                lower_q = q.lower().strip()
                if lower_q in cat_lower:
                    category_map[q] = cat_lower[lower_q]
                    logger.info(f"  Mapped query '{q}' → rating category '{cat_lower[lower_q]}'")
                else:
                    logger.info(f"  No rating category match for query '{q}', using global model")
        except Exception as e:
            logger.warning(f"Could not load rating categories: {e}")

    # ---- Step 2 & 3: Retrieve and rate for each query ----
    for query in queries:
        logger.info(f"  Processing query: '{query}'")
        try:
            # Search the corpus
            retrieved = await search_corpus(query, top_k=top_k, method=retrieval_method)
        except Exception as e:
            logger.error(f"  Search failed for '{query}': {e}")
            report["results"].append({
                "query": query,
                "error": f"Search failed: {e}",
                "clauses": [],
            })
            continue

        # For each retrieved document, get the full text and rate it
        rated_clauses = []
        for item in retrieved:
            doc_id = item["doc_id"]
            try:
                # Get full clause text (search_corpus returns truncated)
                doc = await get_document_by_id(doc_id, include_full_text=True)
                if doc.get("error"):
                    logger.warning(f"    Could not fetch full text for {doc_id}: {doc['error']}")
                    continue
                full_text = doc["text"]
            except Exception as e:
                logger.warning(f"    Error fetching {doc_id}: {e}")
                continue

            # Rate the clause
            category_hint = category_map.get(query)  # exact category name if matched, else None
            try:
                rating_result = await rate_clause(full_text, category=category_hint)
            except Exception as e:
                logger.warning(f"    Rating failed for {doc_id}: {e}")
                continue

            rated_clauses.append({
                "doc_id": doc_id,
                "score": item["score"],
                "clause_text": full_text,
                "predicted_rating": rating_result.get("predicted_rating"),
                "stars": _make_stars(rating_result.get("predicted_rating", 0.0)),
            })

        report["results"].append({
            "query": query,
            "rated_clauses": ranked_by_rating(rated_clauses),  # optional sort
        })

    logger.info("✅ Submission pipeline completed")
    return report


def _make_stars(rating: float) -> str:
    """Convert numeric rating to a 5-star string (nearest 0.5)."""
    if rating is None:
        return "☆☆☆☆☆"
    star_count = round(rating * 2) / 2
    stars = ''.join(
        '★' if i < int(star_count) else '½' if i < star_count and star_count % 1 else '☆'
        for i in range(5)
    )
    return stars


def ranked_by_rating(clauses: List[Dict]) -> List[Dict]:
    """Sort clauses by predicted_rating descending (None at end)."""
    return sorted(clauses, key=lambda x: (x.get("predicted_rating") is None, x.get("predicted_rating") or 0), reverse=True)