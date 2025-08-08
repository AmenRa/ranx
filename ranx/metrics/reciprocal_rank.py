from typing import Union

import numpy as np

from ..decorators import maybe_njit
from .common import clean_qrels, fix_k


# LOW LEVEL FUNCTIONS ==========================================================
@maybe_njit(cache=True)
def _reciprocal_rank(qrels, run, k, rel_lvl):
    qrels = clean_qrels(qrels, rel_lvl)
    if len(qrels) == 0:
        return 0.0

    k = fix_k(k, run)

    for i in range(k):
        if run[i, 0] in qrels[:, 0]:
            return 1 / (i + 1)
    return 0.0


# Handle parallel version with conditional compilation
try:
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def _reciprocal_rank_parallel_numba(qrels, run, k, rel_lvl):
        scores = np.zeros((len(qrels)), dtype=np.float64)
        for i in prange(len(qrels)):
            scores[i] = _reciprocal_rank(qrels[i], run[i], k, rel_lvl)
        return scores

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _reciprocal_rank_numpy(qrels, run, k, rel_lvl):
    """NumPy fallback implementation."""
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in range(len(qrels)):
        scores[i] = _reciprocal_rank(qrels[i], run[i], k, rel_lvl)
    return scores


def _reciprocal_rank_parallel(qrels, run, k, rel_lvl):
    """Dispatch to best available implementation."""
    from ..config import use_numba

    if NUMBA_AVAILABLE and use_numba():
        return _reciprocal_rank_parallel_numba(qrels, run, k, rel_lvl)
    else:
        return _reciprocal_rank_numpy(qrels, run, k, rel_lvl)


# HIGH LEVEL FUNCTIONS =========================================================
def reciprocal_rank(
    qrels: Union[np.ndarray, list],
    run: Union[np.ndarray, list],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Reciprocal Rank (at k).

    The Reciprocal Rank is the multiplicative inverse of the rank of the first retrieved relevant document: 1 for first place, 1/2 for second place, 1/3 for third place, and so on.<br />
    If k > 0, only the top-k retrieved documents are considered.

    $$
    Reciprocal Rank = \frac{1}{rank}
    $$

    where,

    - $rank$ is the position of the first retrieved relevant document.

    Args:
        qrels: IDs and relevance scores of _relevant_ documents.

        run: IDs and relevance scores of _retrieved_ documents.

        k (int, optional): This argument is ignored. It was added to standardize metrics' input. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Reciprocal Rank (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _reciprocal_rank_parallel(qrels, run, k, rel_lvl)
