import logging
from src.mcp_insurance.core.pdf_extractor import extract_text_from_pdf, parse_policy_data
from src.mcp_insurance.core.corpus_loader import CorpusLoader
from src.mcp_insurance.core.retriever import Retriever
from src.mcp_insurance.tools.rating import rate_clause
from src.mcp_insurance.data.dataset_paths import DatasetPaths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_submission(file_path: str, top_k: int = 5) -> dict:
    """
    Parse ACORD PDF, retrieve top clauses, and rate them.
    """
    # 1. Extract text and policy fields
    full_text = extract_text_from_pdf(file_path)
    if not full_text:
        return {"error": "Could not extract text from PDF"}

    policy_data = parse_policy_data(full_text)

    # 2. Initialize retriever with corpus (needs BM25 index built)
    paths = DatasetPaths()
    corpus_loader = CorpusLoader(paths.corpus_jsonl).load().build_bm25_index()
    retriever = Retriever(corpus_loader)

    # 3. Search with BM25
    search_results = retriever.search_bm25(full_text, top_k=top_k)

    # 4. Rate each retrieved clause
    clauses = []
    for doc_id, score in search_results:
        clause_text = corpus_loader.get_document(doc_id)
        if not clause_text:
            continue

        rating_result = await rate_clause(clause_text)
        # rating_result contains: predicted_rating, matched_category, stars, etc.
        clauses.append({
            "doc_id": doc_id,
            "text": clause_text,
            "relevance_score": round(score, 2),
            "predicted_rating": rating_result.get("predicted_rating"),
            "stars": rating_result.get("stars"),
            "matched_category": rating_result.get("matched_category"),
        })

    # Sort by rating descending
    clauses.sort(key=lambda c: (c["predicted_rating"] is None, c["predicted_rating"] or 0), reverse=True)

    return {
        "policy_fields": policy_data,
        "clauses": clauses,
    }