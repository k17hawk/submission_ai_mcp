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
