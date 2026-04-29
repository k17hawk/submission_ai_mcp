"""Load qrels (train/valid/test) from TSV or JSONL files."""

def load_qrels(split: str = "train") -> dict:
    """
    Load relevance judgments.
    split: 'train', 'valid', or 'test'
    Returns dict {query_id: {doc_id: relevance}}
    """
    # TODO: implement
    return {}
