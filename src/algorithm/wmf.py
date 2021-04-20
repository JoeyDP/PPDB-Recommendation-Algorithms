from pathlib import Path

from .algorithm import Algorithm

import numpy as np
import scipy.sparse


import implicit


class WMF(Algorithm):
    """ Recommends based on a matrix completion scheme with user and item factors. """
    model: implicit.als.AlternatingLeastSquares

    def __init__(self, alpha: float = 40, num_factors: int = 100, regularization: float = 0.01, iterations: int = 20):
        """
        Initialize the weighted matrix factorization algorithm.
        :param alpha: Alpha parameter for the confidence of positive samples. Negative samples have conf 1.
        :param num_factors: Dimension of factors used by the user- and item-factors.
        :param regularization: L2 norm regularization parameter on the factors.
        :param iterations: Number of iterations to execute the ALS calculation.
        """
        super().__init__()
        self.alpha = alpha

        self.num_factors = num_factors
        self.regularization = regularization
        self.iterations = iterations

    def create_model(self):
        return implicit.als.AlternatingLeastSquares(
            factors=self.num_factors, iterations=self.iterations, regularization=self.regularization,
            use_gpu=False, use_cg=True, use_native=True
        )

    def fit(self, X: scipy.sparse.csr_matrix) -> 'WMF':
        self.model = self.create_model()
        self.model.fit(self.alpha * X.T)
        return self

    def predict(self, histories: scipy.sparse.csr_matrix) -> np.ndarray:
        predictions = np.zeros(histories.shape)
        for u in range(histories.shape[0]):
            recommendations = self.model.recommend(
                0, histories[u], recalculate_user=True, filter_already_liked_items=False, N=histories.shape[1]
            )
            items, scores = zip(*recommendations)
            predictions[u, items] = scores
        return predictions

    def save(self, path: Path):
        np.save(path, self.model.item_factors)

    def load(self, path: Path) -> 'WMF':
        self.model = self.create_model()
        self.model.item_factors = np.load(path, allow_pickle=False)
        return self
