from .algorithm import Algorithm

import numpy as np
import scipy.sparse


class Random(Algorithm):
    def predict(self, histories: scipy.sparse.csr_matrix) -> np.array:
        return np.random.random(histories.shape)
