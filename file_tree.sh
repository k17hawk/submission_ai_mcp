#!/usr/bin/env bash
set -e  # exit on first error

# Create all directories (including nested ones)
mkdir -p core data models utils scripts tests

# ----------------------------------------------------------------------
# core/ (business logic, no MCP decorators)
# ----------------------------------------------------------------------

# core/__init__.py
cat > core/__init__.py << 'EOF'
"""Core business logic package."""
EOF

# core/pdf_extractor.py
cat > core/pdf_extractor.py << 'EOF'
"""PDF extraction and policy data parsing."""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    # TODO: implement using PyPDF2, pdfplumber, etc.
    return ""

def parse_policy_data(text: str) -> dict:
    """Parse policy text into structured data."""
    # TODO: implement extraction logic (clauses, conditions, etc.)
    return {}
EOF

# core/corpus_loader.py
cat > core/corpus_loader.py << 'EOF'
"""Load JSONL corpus and build search indices (BM25 / embeddings)."""

def load_corpus_jsonl(path: str) -> list:
    """Load corpus from JSONL file, return list of dicts."""
    # TODO: implement
    return []

def build_index(corpus: list, method: str = "bm25"):
    """Build BM25 or embedding index for retrieval."""
    # TODO: implement using rank_bm25 or sentence-transformers
    pass
EOF

# core/retriever.py
cat > core/retriever.py << 'EOF'
"""Retrieval and ranking functions."""

def search_index(query: str, index, top_k: int = 10) -> list:
    """Retrieve top-k documents from pre-built index."""
    # TODO: implement
    return []

def rank_documents(query: str, documents: list) -> list:
    """Rerank retrieved documents (e.g., cross-encoder)."""
    # TODO: implement
    return documents
EOF

# core/qrels_loader.py
cat > core/qrels_loader.py << 'EOF'
"""Load qrels (train/valid/test) from TSV or JSONL files."""

def load_qrels(split: str = "train") -> dict:
    """
    Load relevance judgments.
    split: 'train', 'valid', or 'test'
    Returns dict {query_id: {doc_id: relevance}}
    """
    # TODO: implement
    return {}
EOF

# core/clause_ratings.py
cat > core/clause_ratings.py << 'EOF'
"""Load Excel ratings and predict clause ratings."""

def load_excel_ratings(excel_path: str) -> dict:
    """Load clause ratings from an Excel file."""
    # TODO: implement using pandas
    return {}

def predict_rating(clause_text: str, model=None) -> float:
    """Predict rating for a single clause (optional ML model)."""
    # TODO: implement
    return 0.0
EOF

# ----------------------------------------------------------------------
# data/ (data loading and caching)
# ----------------------------------------------------------------------
cat > data/__init__.py << 'EOF'
"""Data handling package."""
EOF

cat > data/dataset_paths.py << 'EOF'
"""Resolve absolute paths to corpus, qrels, Excel ratings."""

def resolve_paths() -> dict:
    """Return dict with keys: corpus, qrels_train, qrels_valid, qrels_test, ratings_excel."""
    # TODO: implement path resolution (environment vars / default locations)
    return {}
EOF

cat > data/preprocess.py << 'EOF'
"""Convert TSV/Excel to internal formats (JSONL, SQLite, etc.)."""

def tsv_to_jsonl(tsv_path: str, jsonl_path: str):
    """Convert TSV file to JSONL format."""
    # TODO: implement
    pass

def excel_to_jsonl(excel_path: str, jsonl_path: str):
    """Convert Excel ratings to JSONL."""
    # TODO: implement
    pass
EOF

# ----------------------------------------------------------------------
# models/ (ML models, e.g., embeddings)
# ----------------------------------------------------------------------
cat > models/__init__.py << 'EOF'
"""ML models package."""
EOF

cat > models/embedding_model.py << 'EOF'
"""Singleton for sentence-transformers / embedding model."""

import threading

class EmbeddingModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        # TODO: load sentence-transformers model once
        self.model = None

    def encode(self, texts: list) -> list:
        """Encode texts into embeddings."""
        # TODO: implement
        return []
EOF

# ----------------------------------------------------------------------
# utils/ (shared helpers)
# ----------------------------------------------------------------------
cat > utils/__init__.py << 'EOF'
"""Utility functions package."""
EOF

cat > utils/file_utils.py << 'EOF'
"""File and path helpers."""

import os

def ensure_absolute_path(path: str) -> str:
    """Convert relative path to absolute using current working directory."""
    return os.path.abspath(path)

def check_file_exists(path: str) -> bool:
    """Return True if file exists and is a regular file."""
    return os.path.isfile(path)
EOF

cat > utils/logging_config.py << 'EOF'
"""Structured logging configuration (e.g., JSON logs)."""

import logging

def setup_logging(level=logging.INFO):
    """Configure logging with structured output."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # TODO: add JSON formatter if needed
EOF

# ----------------------------------------------------------------------
# scripts/ (one-off utilities)
# ----------------------------------------------------------------------
cat > scripts/index_corpus.py << 'EOF'
#!/usr/bin/env python3
"""Pre‑compute BM25/FAISS index on corpus.jsonl."""

def main():
    print("Building index...")
    # TODO: implement indexing

if __name__ == "__main__":
    main()
EOF

cat > scripts/prepare_queries.py << 'EOF'
#!/usr/bin/env python3
"""Load queries.jsonl into SQLite for fast lookup."""

def main():
    print("Preparing queries DB...")
    # TODO: implement

if __name__ == "__main__":
    main()
EOF

cat > scripts/eval_on_qrels.py << 'EOF'
#!/usr/bin/env python3
"""Run offline evaluation (not as MCP tool)."""

def main():
    print("Evaluating retrieval...")
    # TODO: implement

if __name__ == "__main__":
    main()
EOF

# ----------------------------------------------------------------------
# tests/ (unit/integration tests)
# ----------------------------------------------------------------------
cat > tests/test_parsing.py << 'EOF'
import unittest

class TestParsing(unittest.TestCase):
    def test_extract_text(self):
        # TODO
        pass

if __name__ == "__main__":
    unittest.main()
EOF

cat > tests/test_retrieval.py << 'EOF'
import unittest

class TestRetrieval(unittest.TestCase):
    def test_search_index(self):
        # TODO
        pass

if __name__ == "__main__":
    unittest.main()
EOF

cat > tests/test_evaluation.py << 'EOF'
import unittest

class TestEvaluation(unittest.TestCase):
    def test_qrels_loading(self):
        # TODO
        pass

if __name__ == "__main__":
    unittest.main()
EOF
