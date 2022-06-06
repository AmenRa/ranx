from typing import Union

import numba
import numpy as np
from numba import config, njit, prange
from numba.typed import List as TypedList

config.THREADING_LAYER = "workqueue"

from .common import _clean_qrels, fix_k


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _get_hit_list(qrels, run, k):
    qrels = _clean_qrels(qrels.copy())
    k = fix_k(k, run)

    hit_list = np.full(k, 0.0)

    if len(qrels) == 0:
        return hit_list

    max_true_id = np.max(qrels[:, 0])
    min_true_id = np.min(qrels[:, 0])

    for i in range(k):
        if run[i, 0] > max_true_id:
            continue
        if run[i, 0] < min_true_id:
            continue
        for j in range(qrels.shape[0]):
            if run[i, 0] == qrels[j, 0]:
                hit_list[i] = 1.0
                break

    return hit_list


@njit(cache=True, parallel=True)
def _get_hit_list_parallel(qrels, run, k):
    hit_lists = TypedList(
        [np.ones(1, dtype=np.float64) for _ in range(len(qrels))]
    )
    for i in prange(len(qrels)):
        hit_lists[i] = _get_hit_list(qrels[i], run[i], k)
    return hit_lists


# HIGH LEVEL FUNCTIONS =========================================================
def get_hit_lists(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
) -> np.ndarray:
    """Compute the hit lists (at k).

    **Hit list** is a binary list whose element in position p is 1 if the document retrieved in position p is relevant, 0 otherwise.<br />
    If k > 0, only the top-k retrieved documents are considered.

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

    Returns:
        Hit lists (at k).
    """

    assert k >= 0, "k must be grater or equal to 0"

    return _get_hit_list_parallel(qrels, run, k)
