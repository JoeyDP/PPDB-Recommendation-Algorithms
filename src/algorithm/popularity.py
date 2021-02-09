from pathlib import Path

from .algorithm import Algorithm

import numpy as np
import scipy.sparse


class Popularity(Algorithm):
    item_counts: np.array

    def __init__(self, k: int = 200):
        super().__init__()
        self.k = k

    def fit(self, X: scipy.sparse.csr_matrix) -> 'Popularity':
        self.item_counts = np.asarray(X.sum(axis=0)).flatten() / X.shape[0]
        return self

    def predict(self, histories: scipy.sparse.csr_matrix) -> np.array:
        predictions = self.item_counts[np.newaxis, :].repeat(histories.shape[0], axis=0)
        return predictions

    def save(self, path: Path):
        np.save(path, self.item_counts)

    def load(self, path: Path) -> 'Popularity':
        self.item_counts = np.load(path)
        return self
