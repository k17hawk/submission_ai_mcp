# core/retriever.py
"""
Retrieve and rank documents given a query.
"""

from typing import List, Tuple, Optional
import numpy as np

from .corpus_loader import CorpusLoader


class Retriever:
    def __init__(self, corpus_loader: CorpusLoader):
        self.corpus = corpus_loader

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return list of (doc_id, bm25_score) sorted descending."""
        if self.corpus.bm25 is None:
            raise ValueError("BM25 index not built. Call build_bm25_index() first.")
        tokenized_query = self.corpus._tokenize(query)
        scores = self.corpus.bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.corpus.doc_ids[i], scores[i]) for i in top_indices]
        return results

    def search_embedding(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using cosine similarity on embeddings."""
        if self.corpus.embeddings is None:
            raise ValueError("Embedding index not built. Call build_embedding_index() first.")
        query_emb = self.corpus.embedding_model.encode([query])[0]
        # Cosine similarity
        similarities = np.dot(self.corpus.embeddings, query_emb) / (
            np.linalg.norm(self.corpus.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.corpus.doc_ids[i], similarities[i]) for i in top_indices]
        return results

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Tuple[str, float]]:
        """Combine BM25 and embedding scores with weight alpha (alpha * bm25_norm + (1-alpha) * emb_sim)."""
        bm25_results = dict(self.search_bm25(query, top_k=len(self.corpus.doc_ids)))
        emb_results = dict(self.search_embedding(query, top_k=len(self.corpus.doc_ids)))
        # Normalize scores to [0,1] per method
        if bm25_results:
            max_bm25 = max(bm25_results.values())
            min_bm25 = min(bm25_results.values())
            if max_bm25 != min_bm25:
                bm25_norm = {doc: (score - min_bm25) / (max_bm25 - min_bm25) for doc, score in bm25_results.items()}
            else:
                bm25_norm = {doc: 0.5 for doc in bm25_results}
        else:
            bm25_norm = {}
        if emb_results:
            max_emb = max(emb_results.values())
            min_emb = min(emb_results.values())
            if max_emb != min_emb:
                emb_norm = {doc: (score - min_emb) / (max_emb - min_emb) for doc, score in emb_results.items()}
            else:
                emb_norm = {doc: 0.5 for doc in emb_results}
        else:
            emb_norm = {}
        all_docs = set(bm25_norm.keys()) | set(emb_norm.keys())
        combined = {}
        for doc in all_docs:
            b = bm25_norm.get(doc, 0)
            e = emb_norm.get(doc, 0)
            combined[doc] = alpha * b + (1 - alpha) * e
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs

    def rank_documents(self, query: str, doc_ids: List[str], method: str = "bm25") -> List[Tuple[str, float]]:
        """Rank a given list of document IDs by relevance to query."""
        # For efficiency, we can score only those documents.
        if method == "bm25":
            if self.corpus.bm25 is None:
                raise ValueError("BM25 not built.")
            # Filter corpus texts to only those ids
            indices = [self.corpus.doc_ids.index(did) for did in doc_ids if did in self.corpus.doc_ids]
            subset_texts = [self.corpus.texts[i] for i in indices]
            tokenized_subset = [self.corpus._tokenize(t) for t in subset_texts]
            # Build temporary BM25 on subset
            from rank_bm25 import BM25Okapi
            subset_bm25 = BM25Okapi(tokenized_subset)
            tokenized_query = self.corpus._tokenize(query)
            scores = subset_bm25.get_scores(tokenized_query)
            results = [(doc_ids[i], scores[i]) for i in range(len(doc_ids))]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        elif method == "embedding" and self.corpus.embeddings is not None:
            query_emb = self.corpus.embedding_model.encode([query])[0]
            indices = [self.corpus.doc_ids.index(did) for did in doc_ids if did in self.corpus.doc_ids]
            subset_embs = self.corpus.embeddings[indices]
            similarities = np.dot(subset_embs, query_emb) / (
                np.linalg.norm(subset_embs, axis=1) * np.linalg.norm(query_emb)
            )
            results = [(doc_ids[i], similarities[i]) for i in range(len(doc_ids))]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        else:
            raise ValueError("Method must be 'bm25' or 'embedding' and appropriate index must be built.")