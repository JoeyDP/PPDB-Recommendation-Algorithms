from .algorithm import Algorithm

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNN(Algorithm):
    similarity_matrix_: scipy.sparse.csr_matrix

    def __init__(self, k: int = 200, normalize=False):
        super().__init__()
        self.k = k                      # k=0 means no max neighbors
        self.normalize = normalize

    def fit(self, X: scipy.sparse.csr_matrix):
        item_cosine_similarities_ = cosine_similarity(X.T, dense_output=True)

        # Set diagonal to 0, because we don't want to support self similarity
        np.fill_diagonal(item_cosine_similarities_, 0)

        if self.k:
            top_k_per_row = np.argpartition(item_cosine_similarities_, -self.k, axis=1)[:, -self.k:]
            values = np.take_along_axis(item_cosine_similarities_, top_k_per_row, axis=1)

            res = scipy.sparse.lil_matrix(item_cosine_similarities_.shape)
            np.put_along_axis(res, top_k_per_row, values, axis=1)
            item_cosine_similarities_ = res.tocsr()

        if self.normalize:
            # normalize per row
            row_sums = item_cosine_similarities_.sum(axis=1)
            item_cosine_similarities_ = item_cosine_similarities_ / row_sums

        self.similarity_matrix_ = scipy.sparse.csr_matrix(item_cosine_similarities_)

    def predict(self, histories: scipy.sparse.csr_matrix) -> np.array:
        predictions = histories @ self.similarity_matrix_
        return predictions.toarray()