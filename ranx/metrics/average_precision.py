from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"

from .common import _clean_qrels, fix_k


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _average_precision(qrels, run, k):
    qrels = _clean_qrels(qrels.copy())
    if len(qrels) == 0:
        return 0.0

    k = fix_k(k, run)

    hit_list = np.zeros((k), dtype=np.float64)

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

    precision_scores = np.zeros((k), dtype=np.float64)

    for r in range(k):
        if hit_list[r]:
            # Compute precision at k without computing hit list at k again
            # same as _precision(qrels, run, r + 1)
            precision_scores[r] = np.sum(hit_list[: r + 1]) / (r + 1)

    return np.sum(precision_scores) / qrels.shape[0]


@njit(cache=True, parallel=True)
def _average_precision_parallel(qrels, run, k):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _average_precision(qrels[i], run[i], k)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def average_precision(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
) -> np.ndarray:
    r"""Compute Average Precision.

    Average Precision is the average of the Precision scores computed after each relevant document is retrieved.<br />
    If k > 0, only the top-k retrieved documents are considered.

    $$
    \operatorname{Average Precision} = \frac{\sum_r \operatorname{Precision}@r}{R}
    $$

    where,

    - $r$ is the position of a relevant document;
    - $R$ is the total number of relevant documents.

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        k (int, optional): This argument is ignored. It was added to standardize metrics' input. Defaults to 0.

    Returns:
        Average Precision (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _average_precision_parallel(qrels, run, k)
