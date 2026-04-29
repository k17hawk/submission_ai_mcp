# core/corpus_loader.py
"""
Load corpus and build indexes for retrieval.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
import numpy as np

# If you want embeddings, uncomment and install sentence-transformers
# from sentence_transformers import SentenceTransformer


class CorpusLoader:
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.documents: Dict[str, str] = {}          # doc_id -> text
        self.doc_ids: List[str] = []                 # ordered list of ids
        self.texts: List[str] = []                   # ordered list of texts
        self.bm25: Optional[BM25Okapi] = None
        # self.embedding_model = None
        # self.embeddings: Optional[np.ndarray] = None

    def load(self) -> 'CorpusLoader':
        """Load JSONL file into memory."""
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc['_id']
                text = doc['text']
                self.documents[doc_id] = text
                self.doc_ids.append(doc_id)
                self.texts.append(text)
        return self

    def build_bm25_index(self) -> 'CorpusLoader':
        """Build BM25 index from tokenized texts."""
        tokenized_corpus = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, strip punctuation, split on whitespace."""
        import re
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def build_embedding_index(self, model_name: str = 'all-MiniLM-L6-v2') -> 'CorpusLoader':
        """Build dense embeddings using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        self.embedding_model = SentenceTransformer(model_name)
        self.embeddings = self.embedding_model.encode(self.texts, show_progress_bar=True)
        return self

    def get_document(self, doc_id: str) -> Optional[str]:
        return self.documents.get(doc_id)

    def get_all_documents(self) -> Dict[str, str]:
        return self.documents

    def get_document_ids(self) -> List[str]:
        return self.doc_ids