from typing import Tuple

import scipy.sparse
import numpy as np

csr_matrix = scipy.sparse.csr_matrix


def weak_generalization(X: csr_matrix, perc_history: float) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """ Splits interaction matrix X in three parts: training, val_in and val_out.
    Users in val_in are the same as in training.
    """
    test_user_ids = list()
    val_in = X.copy()
    for u in range(val_in.shape[0]):
        items = val_in[u].nonzero()[1]
        if len(items) < 2:
            continue
        amt_out = int(len(items) * perc_history)
        amt_out = max(1, amt_out)                   # at least one test item required
        amt_out = min(len(items) - 1, amt_out)      # at least one train item required
        items_out = np.random.choice(items, amt_out, replace=False)
        val_in[u, items_out] = 0
        test_user_ids.append(u)

    val_in.eliminate_zeros()

    val_out = X.copy()
    val_out[val_in.astype(bool)] = 0
    val_out.eliminate_zeros()

    return val_in, val_in[test_user_ids], val_out[test_user_ids]
