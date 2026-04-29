# tools/evaluation.py
"""
MCP tools for evaluating retrieval performance using qrels.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

from core.qrels_loader import QrelsLoader
from core.retriever import Retriever
from core.corpus_loader import CorpusLoader

# Global instances (lazy-loaded)
_qrels_loader: Optional[QrelsLoader] = None
_corpus_loader: Optional[CorpusLoader] = None
_retriever: Optional[Retriever] = None


def _ensure_initialized():
    global _qrels_loader, _corpus_loader, _retriever
    if _qrels_loader is None:
        from dataset_paths import DatasetPaths
        paths = DatasetPaths()
        _qrels_loader = QrelsLoader(paths.qrels_dir)
        _corpus_loader = CorpusLoader(paths.corpus_jsonl).load().build_bm25_index()
        _retriever = Retriever(_corpus_loader)


async def evaluate_retrieval(
    split: str = "train",
    top_k: int = 10,
    method: str = "bm25"
) -> Dict[str, Any]:
    """
    MCP Tool: Evaluate retrieval performance on a qrels split.

    Args:
        split: 'train', 'valid', or 'test'.
        top_k: Number of retrieved documents to consider.
        method: Ranking method ("bm25", "embedding", "hybrid").

    Returns:
        Dictionary with metrics: mAP, NDCG@k, recall@k, precision@k.
    """
    _ensure_initialized()
    qrels = _qrels_loader.load_qrels(split)
    queries = _load_queries()  # need to load queries.jsonl for texts

    all_scores = []
    for query_id, rel_dict in qrels.items():
        # Get query text
        query_text = queries.get(query_id)
        if not query_text:
            continue

        # Retrieve
        if method == "bm25":
            ranked = _retriever.search_bm25(query_text, top_k=top_k)
        elif method == "embedding":
            ranked = _retriever.search_embedding(query_text, top_k=top_k)
        else:
            ranked = _retriever.hybrid_search(query_text, top_k=top_k)

        # Build relevance vector for top_k
        y_true = []
        y_pred = []
        for rank, (doc_id, score) in enumerate(ranked):
            rel = rel_dict.get(doc_id, 0)
            y_true.append(rel)
            # For NDCG we need graded relevance; scores from qrels are 1-5 (or 0 if missing)
            y_pred.append(1.0 / (rank + 1) if rank < top_k else 0)  # simple reciprocal

        if len(y_true) == 0:
            continue

        # Compute per-query metrics
        # Precision@k
        prec = sum(1 for rel in y_true if rel > 0) / len(y_true)
        # Recall@k (relative to total relevant docs for this query)
        total_rel = sum(1 for rel in rel_dict.values() if rel > 0)
        recall = sum(1 for rel in y_true if rel > 0) / total_rel if total_rel > 0 else 0
        # NDCG@k
        y_true_graded = np.array([y_true]).reshape(1, -1)
        y_pred_scores = np.array([y_pred]).reshape(1, -1)
        ndcg = ndcg_score(y_true_graded, y_pred_scores, k=top_k)

        all_scores.append({"precision": prec, "recall": recall, "ndcg": ndcg})

    # Aggregate
    avg_precision = np.mean([s["precision"] for s in all_scores])
    avg_recall = np.mean([s["recall"] for s in all_scores])
    avg_ndcg = np.mean([s["ndcg"] for s in all_scores])

    return {
        "split": split,
        "method": method,
        "top_k": top_k,
        "average_precision": float(avg_precision),
        "average_recall": float(avg_recall),
        "average_ndcg": float(avg_ndcg),
        "num_queries": len(all_scores)
    }


def _load_queries() -> Dict[str, str]:
    """Helper: load queries.jsonl into {query_id: text}."""
    import json
    from dataset_paths import DatasetPaths
    paths = DatasetPaths()
    queries = {}
    with open(paths.queries_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']
    return queries