from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"

from .common import clean_qrels, fix_k


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _hit_rate(qrels, run, k, rel_lvl):
    qrels = clean_qrels(qrels, rel_lvl)
    if len(qrels) == 0:
        return 0.0

    k = fix_k(k, run)

    max_true_id = np.max(qrels[:, 0])
    min_true_id = np.min(qrels[:, 0])

    for i in range(k):
        if run[i, 0] > max_true_id:
            continue
        if run[i, 0] < min_true_id:
            continue
        for j in range(qrels.shape[0]):
            if run[i, 0] == qrels[j, 0]:
                return 1.0

    return 0.0


@njit(cache=True, parallel=True)
def _hit_rate_parallel(qrels, run, k, rel_lvl):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _hit_rate(qrels[i], run[i], k, rel_lvl)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def hit_rate(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    """Compute Hit Rate (at k).

    **Hit Rate** is the fraction of queries for which at least one relevant document is retrieved.<br />
    If k > 0, only the top-k retrieved documents are considered.
    Note: it is equivalent to `success` from [trec_eval](https://github.com/usnistgov/trec_eval).

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Hit Rate (at k) scores.
    """

    assert k >= 0, "k must be grater or equal to 0"

    return _hit_rate_parallel(qrels, run, k, rel_lvl)
