from pathlib import Path

from abc import ABC, abstractmethod

import scipy.sparse
import numpy as np


class Algorithm(ABC):
    def fit(self, X: scipy.sparse.csr_matrix) -> 'Algorithm':
        return self

    @abstractmethod
    def predict(self, histories: scipy.sparse.csr_matrix) -> np.array:
        pass

    @abstractmethod
    def save(self, path: Path):
        pass

    @abstractmethod
    def load(self, path: Path) -> 'Algorithm':
        pass
