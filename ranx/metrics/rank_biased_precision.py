from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"

from .get_hit_lists import _get_hit_list


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _rank_biased_precision(qrels, run, p):
    hit_list = _get_hit_list(qrels, run, k=0)
    p_values = p ** np.arange(len(hit_list))
    p_values_sum = sum(hit_list * p_values)

    return (1 - p) * p_values_sum


@njit(cache=True, parallel=True)
def _rank_biased_precision_parallel(qrels, run, p):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _rank_biased_precision(qrels[i], run[i], p)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def rank_biased_precision(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    p: float,
) -> np.ndarray:
    r"""Compute Rank-biased Precision (RBP).

    It is defined as:

    $$
    \operatorname{RBP} = (1 - p) \cdot \sum_{i=1}^{d}{r_i \cdot p^{i - 1}}
    $$

    where,

    - $p$ is the persistence value;
    - $r_i$ is either 0 or 1, whether the $i$-th ranked document is non-relevant or relevant, repsectively.

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        p (float): Persistence value.

    Returns:
        Rank-biased Precision scores.
    """

    return _rank_biased_precision_parallel(qrels, run, p)
