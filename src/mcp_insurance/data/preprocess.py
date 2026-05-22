"""
One-time preprocessing of corpus, queries, and embeddings.
Run this manually when data changes, not as part of server startup.
"""

import json
import pickle
from pathlib import Path

from dataset_paths import DatasetPaths
from core.corpus_loader import CorpusLoader
from core.retriever import Retriever
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def preprocess_and_cache_corpus(cache_dir: Path = Path(".cache")):
    """Tokenize corpus text and optionally save BM25 index or embeddings."""
    cache_dir.mkdir(exist_ok=True)
    paths = DatasetPaths()

    print("Loading corpus...")
    loader = CorpusLoader(paths.corpus_jsonl).load()
    
    cleaned_texts = []
    for doc_id, text in loader.documents.items():
        text = " ".join(text.split())
        cleaned_texts.append(text)
    loader.texts = cleaned_texts
    
    tokenized = [loader._tokenize(t) for t in loader.texts]
    with open(cache_dir / "tokenized_corpus.pkl", "wb") as f:
        pickle.dump(tokenized, f)
    print(f"Saved tokenized corpus to {cache_dir / 'tokenized_corpus.pkl'}")
    
    bm25 = BM25Okapi(tokenized)
    with open(cache_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("Saved BM25 index.")
    
    try:
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(loader.texts, show_progress_bar=True)
        with open(cache_dir / "embeddings.npy", "wb") as f:
            import numpy as np
            np.save(f, embeddings)
        print("Saved embeddings.")
    except ImportError:
        print("sentence-transformers not installed; skipping embeddings.")


def preprocess_queries():
    """Extract query variants from TSV and save as JSON."""
    import pandas as pd
    paths = DatasetPaths()
    df = pd.read_csv(paths.query_tsv, sep='\t')
    variants = {}
    for _, row in df.iterrows():
        qid = row['query_id']
        variants[qid] = {
            'short': row.get('short', ''),
            'medium': row.get('medium', ''),
            'long': row.get('long', '')
        }
    output_path = Path(".cache") / "query_variants.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(variants, f, indent=2)
    print(f"Saved query variants to {output_path}")


if __name__ == "__main__":
    preprocess_and_cache_corpus()
    preprocess_queries()