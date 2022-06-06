import numpy as np
from numba import config, njit

config.THREADING_LAYER = "workqueue"


@njit(cache=True)
def _clean_qrels(qrels):
    return qrels[np.nonzero(qrels[:, 1])]


@njit(cache=True)
def fix_k(k, run):
    if k == 0 or k > run.shape[0]:
        return run.shape[0]
    else:
        return k
