# tools/retrieval.py
"""
MCP tools for retrieving insurance clauses from the corpus.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from core.corpus_loader import CorpusLoader
from core.retriever import Retriever

_corpus_loader: Optional[CorpusLoader] = None
_retriever: Optional[Retriever] = None


def _ensure_initialized():
    """Lazy-load the corpus and build BM25 index."""
    global _corpus_loader, _retriever
    if _corpus_loader is None:
        from dataset_paths import DatasetPaths
        paths = DatasetPaths()
        _corpus_loader = CorpusLoader(paths.corpus_jsonl).load().build_bm25_index()
        _retriever = Retriever(_corpus_loader)


async def search_corpus(query: str, top_k: int = 10, method: str = "bm25") -> List[Dict[str, Any]]:
    """
    MCP Tool: Search the insurance clause corpus.

    Args:
        query: Search query text.
        top_k: Number of top results to return.
        method: Ranking method: "bm25", "embedding", or "hybrid".

    Returns:
        List of dicts, each with "doc_id", "score", and "text".
    """
    _ensure_initialized()
    if method == "bm25":
        results = _retriever.search_bm25(query, top_k)
    elif method == "embedding":
        results = _retriever.search_embedding(query, top_k)
    elif method == "hybrid":
        results = _retriever.hybrid_search(query, top_k)
    else:
        raise ValueError(f"Unknown method: {method}")

    output = []
    for doc_id, score in results:
        text = _corpus_loader.get_document(doc_id)
        output.append({
            "doc_id": doc_id,
            "score": float(score),
            "text": text[:500] + "..." if text and len(text) > 500 else text
        })
    return output


async def get_document_by_id(doc_id: str, include_full_text: bool = True) -> Dict[str, Any]:
    """
    MCP Tool: Retrieve a single document by its ID.

    Args:
        doc_id: The document ID (e.g., "a5a68dbd19").
        include_full_text: If True, include the full text; otherwise just metadata.

    Returns:
        Dictionary with doc_id and text (if requested), or error.
    """
    _ensure_initialized()
    text = _corpus_loader.get_document(doc_id)
    if text is None:
        return {"error": f"Document ID not found: {doc_id}"}
    result = {"doc_id": doc_id}
    if include_full_text:
        result["text"] = text
    else:
        result["text_preview"] = text[:500] + "..." if len(text) > 500 else text
    return result