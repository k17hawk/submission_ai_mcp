from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DatasetPaths:
    def __init__(self, root_dir=None):
        if root_dir is None:
            root_dir = Path(__file__).parent.parent
        self.root = Path(root_dir)
        logger.info(f"DatasetPaths initialized with root directory: {self.root}")
        self.dataset = self.root / "dataset"

    @property
    def corpus_jsonl(self) -> Path:
        logger.info(f"Accessing corpus JSONL path: {self.dataset / 'corpus.jsonl'}")
        return self.dataset / "corpus.jsonl"

    @property
    def queries_jsonl(self) -> Path:
        logger.info(f"Accessing queries JSONL path: {self.dataset / 'queries.jsonl'}")
        return self.dataset / "queries.jsonl"

    @property
    def qrels_dir(self) -> Path:
        logger.info(f"Accessing qrels directory path: {self.dataset / 'qrels'}")
        return self.dataset / "qrels"

    def get_qrels(self, split: str = "train") -> Path:
        """Get qrels file for train/test/valid split."""
        logger.info(f"Accessing qrels file for split '{split}': {self.qrels_dir / f'{split}.tsv'}")
        return self.qrels_dir / f"{split}.tsv"

    @property
    def rating_excel(self) -> Path:
        logger.info(f"Accessing rating Excel path: {self.dataset / 'supplemental materials' / 'ACORD 2-5 Star Clause Pairs.xlsx'}") 
        return self.dataset / "supplemental materials" / "ACORD 2-5 Star Clause Pairs.xlsx"

    @property
    def query_tsv(self) -> Path:
        logger.info(f"Accessing query variants TSV path: {self.dataset / 'supplemental materials' / 'acord query (short_medium_long).tsv'}")    
        return self.dataset / "supplemental materials" / "acord query (short_medium_long).tsv"