import faiss
import numpy as np

from . import config


class FaissStore:
    """HNSW index wrapped with explicit IDs (matching SQLite's faiss_id
    column). HNSW needs no separate training pass -- unlike IVF -- which
    keeps indexing simple while still scaling to millions of vectors."""

    def __init__(self, dim):
        self.dim = dim
        self.path = config.FAISS_PATH
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
        else:
            base = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = 80
            base.hnsw.efSearch = 64
            self.index = faiss.IndexIDMap2(base)

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        self.index.add_with_ids(vectors, ids)

    def search(self, vector: np.ndarray, k: int):
        vector = vector.reshape(1, -1)
        scores, ids = self.index.search(vector, k)
        return scores[0], ids[0]

    def save(self):
        faiss.write_index(self.index, str(self.path))

    def ntotal(self):
        return self.index.ntotal
