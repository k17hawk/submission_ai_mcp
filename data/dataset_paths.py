from pathlib import Path

class DatasetPaths:
    def __init__(self, root_dir=None):
        if root_dir is None:
            root_dir = Path(__file__).parent.parent
        self.root = Path(root_dir)
        self.dataset = self.root / "dataset"

    @property
    def corpus_jsonl(self) -> Path:
        return self.dataset / "corpus.jsonl"

    @property
    def queries_jsonl(self) -> Path:
        return self.dataset / "queries.jsonl"

    @property
    def qrels_dir(self) -> Path:
        return self.dataset / "qrels"

    def get_qrels(self, split: str = "train") -> Path:
        """Get qrels file for train/test/valid split."""
        return self.qrels_dir / f"{split}.tsv"

    @property
    def rating_excel(self) -> Path:
        return self.dataset / "supplemental materials" / "ACORD 2-5 Star Clause Pairs.xlsx"

    @property
    def query_tsv(self) -> Path:
        return self.dataset / "supplemental materials" / "acord query (short_medium_long).tsv"