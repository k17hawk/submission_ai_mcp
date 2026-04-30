"""
Load relevance judgments (qrels) from TSV files.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional


class QrelsLoader:
    def __init__(self, qrels_dir: Path):
        self.qrels_dir = qrels_dir

    def load_qrels(self, split: str = "train") -> Dict[str, Dict[str, int]]:
        """
        Returns {query_id: {doc_id: relevance_score}}
        """
        file_path = self.qrels_dir / f"{split}.tsv"
        if not file_path.exists():
            raise FileNotFoundError(f"Qrels file not found: {file_path}")
        qrels = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            # columns: query-id, corpus-id, score
            for row in reader:
                qid = row['query-id']
                doc_id = row['corpus-id']
                score = int(row['score'])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = score
        return qrels

    def load_all_splits(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Return {'train': ..., 'valid': ..., 'test': ...}"""
        return {split: self.load_qrels(split) for split in ['train', 'valid', 'test']}

    def get_relevant_docs(self, qrels: Dict[str, Dict[str, int]], query_id: str, min_score: int = 1) -> List[str]:
        """Return list of doc_ids that have relevance score >= min_score for a given query."""
        qrel_dict = qrels.get(query_id, {})
        return [doc_id for doc_id, score in qrel_dict.items() if score >= min_score]