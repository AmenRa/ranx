from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"
from .common import clean_qrels


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _interpolated_precision(qrels, run, rel_lvl):
    qrels = clean_qrels(qrels, rel_lvl)
    if len(qrels) == 0:
        return np.zeros(11)

    # Compute cutoffs ----------------------------------------------------------
    cutoff_percents = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    cutoffs = np.empty(cutoff_percents.shape[0], dtype=np.int64)

    # Transform percentage-based cutoffs to num of relevants
    for i in range(cutoffs.shape[0]):
        cutoffs[i] = int(cutoff_percents[i] * qrels.shape[0] + 0.9)

    # Find relevants -----------------------------------------------------------
    k = run.shape[0]
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

    # Compute precision scores -------------------------------------------------
    precision_scores = np.zeros((k + 1), dtype=np.float64)

    for r in range(k):
        if hit_list[r]:
            # Compute precision at k without computing hit list at k again
            # same as _precision(qrels, run, r + 1)
            precision_scores[r + 1] = np.sum(hit_list[: r + 1]) / (r + 1)

    precision_scores[0] = max(precision_scores)

    precision_scores = precision_scores[np.nonzero(precision_scores)]

    # Do interpolation ---------------------------------------------------------
    interpolated_precision_scores = np.empty(cutoffs.shape[0])
    for i in range(cutoffs.shape[0]):
        cutoff = cutoffs[i]
        if cutoff < precision_scores.shape[0]:
            interpolated_precision_scores[i] = max(precision_scores[cutoff:])
        else:
            interpolated_precision_scores[i] = 0.0

    return interpolated_precision_scores


@njit(cache=True, parallel=True)
def _interpolated_precision_parallel(qrels, run, rel_lvl):
    scores = np.zeros((len(qrels), 11), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _interpolated_precision(qrels[i], run[i], rel_lvl)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def interpolated_precision_at_recall(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Interpolated Precision at Recall.

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Interpolated Precision scores.

    """

    return _interpolated_precision_parallel(qrels, run, rel_lvl)
