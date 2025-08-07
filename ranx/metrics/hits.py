from typing import Union

import numpy as np

from ..decorators import maybe_njit
from .common import clean_qrels, fix_k


# LOW LEVEL FUNCTIONS ==========================================================
@maybe_njit(cache=True)
def _hits(qrels, run, k, rel_lvl):
    qrels = clean_qrels(qrels, rel_lvl)
    if len(qrels) == 0:
        return 0.0

    k = fix_k(k, run)

    max_true_id = np.max(qrels[:, 0])
    min_true_id = np.min(qrels[:, 0])

    hits = 0.0

    for i in range(k):
        if run[i, 0] > max_true_id:
            continue
        if run[i, 0] < min_true_id:
            continue
        for j in range(qrels.shape[0]):
            if run[i, 0] == qrels[j, 0]:
                hits += 1.0
                break

    return hits


# Handle parallel version with conditional compilation
try:
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def _hits_parallel_numba(qrels, run, k, rel_lvl):
        scores = np.zeros((len(qrels)), dtype=np.float64)
        for i in prange(len(qrels)):
            scores[i] = _hits(qrels[i], run[i], k, rel_lvl)
        return scores

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _hits_parallel_numpy(qrels, run, k, rel_lvl):
    """NumPy fallback implementation."""
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in range(len(qrels)):
        scores[i] = _hits(qrels[i], run[i], k, rel_lvl)
    return scores


def _hits_parallel(qrels, run, k, rel_lvl):
    """Dispatch to best available implementation."""
    from ..config import use_numba

    if NUMBA_AVAILABLE and use_numba():
        return _hits_parallel_numba(qrels, run, k, rel_lvl)
    else:
        return _hits_parallel_numpy(qrels, run, k, rel_lvl)


# HIGH LEVEL FUNCTIONS =========================================================
def hits(
    qrels: Union[np.ndarray, list],
    run: Union[np.ndarray, list],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    """Compute Hits (at k).

    **Hits** is the number of relevant documents retrieved.<br />
    If k > 0, only the top-k retrieved documents are considered.

    Args:
        qrels: IDs and relevance scores of _relevant_ documents.

        run: IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Hits (at k) scores.
    """

    assert k >= 0, "k must be grater or equal to 0"

    return _hits_parallel(qrels, run, k, rel_lvl)
