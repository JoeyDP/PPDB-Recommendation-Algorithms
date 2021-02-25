from typing import Tuple

import scipy.sparse
import numpy as np

csr_matrix = scipy.sparse.csr_matrix


def strong_generalization(X: csr_matrix, test_users: int, perc_history: float) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """ Splits interaction matrix X in three parts: training, val_in and val_out.
    Users in training and validation are disjoint.
    """
    users = X.shape[0]
    assert users > test_users, "There should be at least one train user left"

    test_user_ids = np.random.choice(np.arange(users), test_users, replace=False)
    train, val = X[~test_user_ids], X[test_user_ids]

    val_in = val.copy()
    for u in range(val_in.shape[0]):
        items = val_in[u].nonzero()[1]
        amt_out = max(1, int(len(items) * perc_history))        # at least one test item required
        items_out = np.random.choice(items, amt_out, replace=False)
        val_in[u, items_out] = 0

    val_out = val
    val_out[val_in.astype(bool)] = 0

    return train, val_in, val_out
