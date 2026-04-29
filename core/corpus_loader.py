"""Load JSONL corpus and build search indices (BM25 / embeddings)."""

def load_corpus_jsonl(path: str) -> list:
    """Load corpus from JSONL file, return list of dicts."""
    # TODO: implement
    return []

def build_index(corpus: list, method: str = "bm25"):
    """Build BM25 or embedding index for retrieval."""
    # TODO: implement using rank_bm25 or sentence-transformers
    pass
