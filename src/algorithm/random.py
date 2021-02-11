from pathlib import Path

from .algorithm import Algorithm

import numpy as np
import scipy.sparse


class Random(Algorithm):
    def predict(self, histories: scipy.sparse.csr_matrix) -> np.ndarray:
        return np.random.random(histories.shape)

    def save(self, path: Path):
        pass

    def load(self, path: Path) -> 'Random':
        return self
