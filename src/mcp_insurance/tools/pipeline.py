# tools/pipeline.py
import logging
from typing import Dict, Any, List, Optional

from src.mcp_insurance.tools.parsing import parse_acord_submission
from src.mcp_insurance.tools.retrieval import search_corpus, get_document_by_id
from src.mcp_insurance.tools.rating import rate_clause, get_available_categories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_submission(
    pdf_path: str,
    queries: Optional[List[str]] = None,
    top_k: int = 5,
    retrieval_method: str = "bm25",
    map_query_to_rating_category: bool = True,
    deduplicate_across_queries: bool = False,
) -> Dict[str, Any]:
    """
    Parse ACORD PDF, detect clause topics, retrieve similar clauses, and rate them.

    Args:
        pdf_path: Path to the ACORD PDF file.
        queries: Optional list of clause categories to search for.
                 If None, auto-detects from PDF content.
        top_k: Number of top documents to retrieve per query.
        retrieval_method: "bm25", "embedding", or "hybrid".
        map_query_to_rating_category: Map detected queries to known rating categories.
        deduplicate_across_queries: If True, each doc_id appears only under its
                                    highest-scoring query.

    Returns:
        Dictionary with policy_data, results list, and optional error.
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
    full_text = parse_result.get("text", "")

    if not full_text.strip():
        report["error"] = "PDF extracted text is empty"
        return report

    # ---- Step 2: Detect relevant clause categories from the PDF text ----
    if queries is None:
        queries = _detect_categories_from_text(full_text)
        logger.info(f"Auto-detected {len(queries)} queries from PDF: {queries}")

    if not queries:
        report["error"] = "No clause categories detected in PDF text"
        return report

    # ---- Prepare rating category lookup ----
    category_map = {}
    if map_query_to_rating_category:
        try:
            available_categories = get_available_categories()
            cat_lower = {c.lower(): c for c in available_categories}
            for q in queries:
                lower_q = q.lower().strip()
                if lower_q in cat_lower:
                    category_map[q] = cat_lower[lower_q]
                else:
                    # Try partial match
                    for cat_key, cat_val in cat_lower.items():
                        if lower_q in cat_key or cat_key in lower_q:
                            category_map[q] = cat_val
                            break
            logger.info(f"Category mapping: {category_map}")
        except Exception as e:
            logger.warning(f"Could not load rating categories: {e}")

    # ---- Step 3: Retrieve and rate for each query ----
    for query in queries:
        logger.info(f"  Processing query: '{query}'")
        try:
            retrieved = await search_corpus(query, top_k=top_k, method=retrieval_method)
        except Exception as e:
            logger.error(f"  Search failed for '{query}': {e}")
            report["results"].append({
                "query": query,
                "error": f"Search failed: {e}",
                "rated_clauses": []
            })
            continue

        rated_clauses = []
        for item in retrieved:
            doc_id = item["doc_id"]
            try:
                doc = await get_document_by_id(doc_id, include_full_text=True)
                if doc.get("error"):
                    continue
                full_clause_text = doc["text"]
            except Exception as e:
                logger.warning(f"  Could not get document {doc_id}: {e}")
                continue

            category_hint = category_map.get(query)
            rating_result = await rate_clause(full_clause_text, category=category_hint)

            rated_clauses.append({
                "doc_id": doc_id,
                "score": round(item["score"], 2),
                "clause_text": full_clause_text,
                "predicted_rating": rating_result.get("predicted_rating"),
                "stars": rating_result.get("stars", "☆☆☆☆☆"),
                "category_used": rating_result.get("category_used"),
            })

        # Sort within each query by rating descending
        rated_clauses.sort(
            key=lambda x: (
                x.get("predicted_rating") is None,
                -(x.get("predicted_rating") or 0)
            )
        )

        report["results"].append({
            "query": query,
            "rated_clauses": rated_clauses,
        })

    # ---- Optional: Deduplicate across queries ----
    if deduplicate_across_queries:
        seen_doc_ids = set()
        for result_group in report["results"]:
            unique_clauses = []
            for clause in result_group["rated_clauses"]:
                if clause["doc_id"] not in seen_doc_ids:
                    seen_doc_ids.add(clause["doc_id"])
                    unique_clauses.append(clause)
            result_group["rated_clauses"] = unique_clauses
        logger.info("Deduplicated clauses across queries")

    logger.info(f"✅ Submission pipeline completed — {len(report['results'])} query groups")
    return report


def _detect_categories_from_text(full_text: str) -> List[str]:
    """Detect which clause categories are present in the PDF text."""
    text_lower = full_text.lower()
    detected = []

    # Category detection rules based on keywords in the text
    detection_rules = {
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
            "trademark", "trade secret", "unauthorized use of intellectual property",
            "ip infringement"
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

    for category, keywords in detection_rules.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(category)
                logger.info(f"  Detected category '{category}' via keyword '{keyword}'")
                break

    # Remove duplicates while preserving order
    seen = set()
    unique_detected = []
    for cat in detected:
        if cat not in seen:
            seen.add(cat)
            unique_detected.append(cat)

    logger.info(f"Detected {len(unique_detected)} categories: {unique_detected}")
    return unique_detected