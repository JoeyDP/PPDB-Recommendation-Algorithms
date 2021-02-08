from .algorithm import Algorithm

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity


class EASE(Algorithm):
    similarity_matrix_: scipy.sparse.csr_matrix

    def __init__(self, l2: float = 200):
        super().__init__()
        self.l2 = l2

    def fit(self, X: scipy.sparse.csr_matrix):
        # Compute P
        XTX = (X.T @ X).toarray()
        P = np.linalg.inv(XTX + self.l2 * np.identity((X.shape[1]), dtype=np.float32))

        # Compute B
        B = np.identity(X.shape[1]) - P @ np.diag(1.0 / np.diag(P))
        B[np.diag_indices(B.shape[0])] = 0.0

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)
        print(self.similarity_matrix_)

    def predict(self, histories: scipy.sparse.csr_matrix) -> np.array:
        predictions = histories @ self.similarity_matrix_
        return predictions.toarray()
