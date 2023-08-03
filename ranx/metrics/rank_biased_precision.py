from typing import Union

import numba
import numpy as np
from numba import njit, prange

from .common import clean_qrels


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _rank_biased_precision(qrels, run, p, rel_lvl):
    qrels = clean_qrels(qrels, rel_lvl)
    if len(qrels) == 0:
        return 0.0

    max_true_id = np.max(qrels[:, 0])
    min_true_id = np.min(qrels[:, 0])

    weighted_hit_list = np.zeros((len(run)), dtype=np.float64)

    for i in range(len(run)):
        if run[i, 0] > max_true_id:
            continue
        if run[i, 0] < min_true_id:
            continue
        for j in range(qrels.shape[0]):
            if run[i, 0] == qrels[j, 0]:
                weighted_hit_list[i] = qrels[j, 1]
                break

    p_values = p ** np.arange(len(weighted_hit_list))
    p_values_sum = sum(weighted_hit_list * p_values)

    return (1 - p) * p_values_sum


@njit(cache=True, parallel=True)
def _rank_biased_precision_parallel(qrels, run, p, rel_lvl):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _rank_biased_precision(qrels[i], run[i], p, rel_lvl)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def rank_biased_precision(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    p: float,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Rank-biased Precision (RBP).

    It is defined as:

    $$
    \operatorname{RBP} = (1 - p) \cdot \sum_{i=1}^{d}{r_i \cdot p^{i - 1}}
    $$

    where,

    - $p$ is the persistence value;
    - $r_i$ is either 0 or 1, whether the $i$-th ranked document is non-relevant or relevant, respectively.

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        p (float): Persistence value.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Rank-biased Precision scores.
    """

    return _rank_biased_precision_parallel(qrels, run, p, rel_lvl)
