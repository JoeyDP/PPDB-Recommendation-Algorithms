from pathlib import Path

from abc import ABC, abstractmethod

import scipy.sparse
import numpy as np


class Algorithm(ABC):
    """ Base class for recommendation algorithms. """

    def fit(self, X: scipy.sparse.csr_matrix) -> 'Algorithm':
        """ Trains the model with X as training data. Returns self. """
        return self

    @abstractmethod
    def predict(self, histories: scipy.sparse.csr_matrix) -> np.ndarray:
        """ Calculates scores per user-item pair for the given user histories. """
        pass

    @abstractmethod
    def save(self, path: Path):
        """ Save the model to file. """
        pass

    @abstractmethod
    def load(self, path: Path) -> 'Algorithm':
        """ Load the model from file and return self. """
        pass
