from pathlib import Path

from .algorithm import Algorithm

import numpy as np
import scipy.sparse


class EASE(Algorithm):
    """ The state of the art EASE model that learns based on a constrained least squares auto-regression.
    Optimizes \hat{B} = argmin_B ||X - XB||^2_F + l2 * ||B||^2_F
    s.t. diag(B) = 0
    """
    similarity_matrix_: scipy.sparse.csr_matrix

    def __init__(self, l2: float = 200):
        """
        :param l2: l2 norm regularization on B.
        """
        super().__init__()
        self.l2 = l2

    def fit(self, X: scipy.sparse.csr_matrix) -> 'EASE':
        # Compute P
        X = X.astype(np.int32)

        XTX = (X.T @ X).toarray()
        P = np.linalg.inv(XTX + self.l2 * scipy.sparse.identity((X.shape[1]), dtype=np.int32))
        del XTX

        # Compute B
        B = scipy.sparse.identity(X.shape[1]) - P @ scipy.sparse.diags(1.0 / np.diag(P))
        np.fill_diagonal(B, 0)

        self.similarity_matrix_ = B
        return self

    def predict(self, histories: scipy.sparse.csr_matrix) -> np.ndarray:
        predictions = histories @ self.similarity_matrix_
        return predictions

    def save(self, path: Path):
        np.savez_compressed(path, B=self.similarity_matrix_)

    def load(self, path: Path) -> 'EASE':
        self.similarity_matrix_ = np.load(path)['B']
        return self
