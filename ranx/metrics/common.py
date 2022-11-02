import numpy as np
from numba import config, njit

config.THREADING_LAYER = "workqueue"


@njit(cache=True)
def clean_qrels(qrels, rel_lvl):
    return qrels[np.argwhere(qrels[:, 1] >= rel_lvl).flatten()]


@njit(cache=True)
def fix_k(k, run):
    return run.shape[0] if k == 0 or k > run.shape[0] else k
