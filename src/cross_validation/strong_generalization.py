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

    active_users = np.array(list(set(X.nonzero()[0])))
    test_user_ids = np.random.choice(active_users, test_users, replace=False)
    test_user_mask = np.zeros(users, dtype=bool)
    test_user_mask[test_user_ids] = 1
    train, val = X[~test_user_mask], X[test_user_mask]
    train.eliminate_zeros()

    # print("train/val users (incl zero)", train.shape[0], "/", val.shape[0])
    # print("train/val users (excl zero)", len(set(train.nonzero()[0])), "/", len(set(val.nonzero()[0])))

    val_in = val.copy()
    for u in range(val_in.shape[0]):
        items = val_in[u].nonzero()[1]
        amt_out = int(len(items) * perc_history)
        amt_out = max(1, amt_out)                   # at least one test item required
        amt_out = min(len(items) - 1, amt_out)      # at least one train item required
        items_out = np.random.choice(items, amt_out, replace=False)

        # print("user", u, "items in/total", len(items) - len(items_out), len(items), f"({(len(items) - len(items_out)) / len(items)})")
        val_in[u, items_out] = 0

    val_in.eliminate_zeros()

    val_out = val
    val_out[val_in.astype(bool)] = 0
    val_out.eliminate_zeros()

    return train, val_in, val_out
