from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"

from .get_hit_lists import _get_hit_list
from .get_unjudged_lists import _get_unjudged_list


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _bpref(qrels, run, rel_lvl):
    n_rels = sum(qrels[:, 1] >= rel_lvl)
    n_non_rels = sum(qrels[:, 1] < rel_lvl)

    hit_list = _get_hit_list(qrels, run, k=0, rel_lvl=rel_lvl)
    unjudged_list = _get_unjudged_list(qrels, run, k=0)

    # Remove unjudged from hit_list
    hit_list = hit_list[unjudged_list != 1]

    return (
        np.sum(
            1.0
            - np.minimum(np.cumsum(hit_list == 0)[hit_list == 1], n_rels)
            / min(n_rels, n_non_rels)
        )
        / n_rels
    )


@njit(cache=True, parallel=True)
def _bpref_parallel(qrels, run, rel_lvl):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _bpref(qrels[i], run[i], rel_lvl)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def bpref(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Bpref.

    **Bpref** is designed for situations where relevance judgments are known to be incomplete. It is defined as:

    $$
    \operatorname{bpref}=\frac{1}{R}\sum_r{1 - \frac{|n ranked higher than r|}{R}}
    $$

    where,

    - $r$ is a relevant document;
    - $n$ is a member of the first R judged nonrelevant documents as retrieved by the system;
    - $R$ is the number of relevant documents.

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        k (int, optional): This argument is ignored. It was added to standardize metrics' input. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Bpref scores.
    """

    return _bpref_parallel(qrels, run, rel_lvl)
