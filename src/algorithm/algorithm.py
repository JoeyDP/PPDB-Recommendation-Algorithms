import scipy.sparse
import numpy as np


class Algorithm:

    def fit(self, X: scipy.sparse.csr_matrix):
        pass

    def predict(self, histories: scipy.sparse.csr_matrix) -> np.array:
        pass
