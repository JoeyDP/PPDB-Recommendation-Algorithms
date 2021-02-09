from .algorithm import Algorithm

import numpy as np
import scipy.sparse


class EASE(Algorithm):
    similarity_matrix_: scipy.sparse.csr_matrix

    def __init__(self, l2: float = 200):
        super().__init__()
        self.l2 = l2

    def fit(self, X: scipy.sparse.csr_matrix):
        # Compute P
        XTX = (X.T @ X).toarray()
        P = np.linalg.inv(XTX + self.l2 * scipy.sparse.identity((X.shape[1]), dtype=np.float32))
        del XTX

        # Compute B
        B = scipy.sparse.identity(X.shape[1]) - P @ scipy.sparse.diags(1.0 / np.diag(P))
        np.fill_diagonal(B, 0)

        self.similarity_matrix_ = scipy.sparse.csr_matrix(B)

    def predict(self, histories: scipy.sparse.csr_matrix) -> np.array:
        predictions = histories @ self.similarity_matrix_
        return predictions.toarray()
