from .algorithm import Algorithm

import numpy as np
import scipy.sparse


class Popularity(Algorithm):
    item_counts: np.array

    def __init__(self, k: int = 200):
        super().__init__()
        self.k = k

    def fit(self, X: scipy.sparse.csr_matrix):
        self.item_counts = np.asarray(X.sum(axis=0)).flatten() / X.shape[0]

    def predict(self, histories: scipy.sparse.csr_matrix) -> np.array:
        predictions = self.item_counts[np.newaxis, :].repeat(histories.shape[0], axis=0)
        return predictions