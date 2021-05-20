import scipy.sparse
import numpy as np

import src.util as util


def recall_k(predictions: np.ndarray, Xval_out: scipy.sparse.csr_matrix, top_k):
    """ Implements a stratified Recall@k that calculates per user:
        #correct / min(k, #user_val_items)
    """
    recommendations, scores = util.predictions_to_recommendations(predictions, top_k=top_k)
    hits = np.take_along_axis(Xval_out, recommendations, axis=1)
    total_hits = np.asarray(hits.sum(axis=1)).flatten()
    best_possible = np.asarray(Xval_out.sum(axis=1)).flatten()
    best_possible[best_possible > top_k] = top_k
    recall_scores = total_hits / best_possible
    return recall_scores
