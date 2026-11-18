"""Customer-level splitting, as described in the manuscript's data protocol.

Every message of one customer lands in exactly one of train / validation / test,
and cross-validation folds inside the training set are grouped by customer too.
Clients are stratified by whether they ever convert, so each split keeps a
similar share of converting customers.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def customer_split(client_ids, y, test_frac=0.15, val_frac_of_train=0.17647, seed=1):
    """Return (idx_train_full, idx_test, idx_train, idx_val) as row indices.

    Defaults give 70/15/15 of customers (0.85 * 0.17647 = 0.15).
    """
    client_ids = np.asarray(client_ids)
    y = np.asarray(y)
    codes, inverse = np.unique(client_ids, return_inverse=True)
    client_pos = np.zeros(len(codes), dtype=np.int8)
    np.maximum.at(client_pos, inverse, (y > 0).astype(np.int8))
    all_clients = np.arange(len(codes))

    tr_full_c, te_c = train_test_split(all_clients, test_size=test_frac,
                                       stratify=client_pos, random_state=seed)
    tr_c, va_c = train_test_split(tr_full_c, test_size=val_frac_of_train,
                                  stratify=client_pos[tr_full_c], random_state=seed)

    def rows(clients):
        mask = np.zeros(len(codes), dtype=bool)
        mask[clients] = True
        return np.flatnonzero(mask[inverse])

    return rows(tr_full_c), rows(te_c), rows(tr_c), rows(va_c)


def grouped_folds(n_splits, seed):
    """Stratified K-fold that keeps each customer inside one fold."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
