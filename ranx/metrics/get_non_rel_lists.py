from typing import Union

import numba
import numpy as np
from numba import config, njit, prange
from numba.typed import List as TypedList

config.THREADING_LAYER = "workqueue"

from ranx.metrics.common import fix_k


@njit(cache=True)
def _non_rel_qrels(qrels):
    return qrels[qrels[:, 1] == 0]


@njit()
def _get_non_rel_list(qrels, run, k):
    qrels = _non_rel_qrels(qrels.copy())
    k = fix_k(k, run)

    non_rel_list = np.full(k, 0.0)

    if len(qrels) == 0:
        return non_rel_list

    max_true_id = np.max(qrels[:, 0])
    min_true_id = np.min(qrels[:, 0])

    for i in range(k):
        if run[i, 0] > max_true_id:
            continue
        if run[i, 0] < min_true_id:
            continue
        for j in range(qrels.shape[0]):
            if run[i, 0] == qrels[j, 0]:
                non_rel_list[i] = 1.0
                break

    return non_rel_list


@njit(parallel=True)
def _get_non_rel_list_parallel(qrels, run, k):
    non_rel_lists = TypedList(
        [np.ones(1, dtype=np.float64) for _ in range(len(qrels))]
    )
    for i in prange(len(qrels)):
        non_rel_lists[i] = _get_non_rel_list(qrels[i], run[i], k)
    return non_rel_lists


def get_non_rel_lists(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
) -> np.ndarray:
    """Compute the non_rel lists (at k).

    **Unjudged list** is a binary list whose element in position p is 1 if the document retrieved in position p is non_rel, 0 otherwise.<br />
    If k > 0, only the top-k retrieved documents are considered.

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

    Returns:
        Unjudged lists (at k).
    """

    assert k >= 0, "k must be grater or equal to 0"

    return _get_non_rel_list_parallel(qrels, run, k)
