from typing import Union

import numpy as np

from ..decorators import maybe_njit
from .common import clean_qrels
from .hits import _hits


# LOW LEVEL FUNCTIONS ==========================================================
@maybe_njit(cache=True)
def _recall(qrels, run, k, rel_lvl):
    qrels = clean_qrels(qrels, rel_lvl)
    if len(qrels) == 0:
        return 0.0

    k = k if k != 0 else run.shape[0]
    if k == 0:
        return 0.0

    return _hits(qrels, run, k, rel_lvl) / qrels.shape[0]


# Handle parallel version with conditional compilation
try:
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def _recall_parallel_numba(qrels, run, k, rel_lvl):
        scores = np.zeros((len(qrels)), dtype=np.float64)
        for i in prange(len(qrels)):
            scores[i] = _recall(qrels[i], run[i], k, rel_lvl)
        return scores

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _recall_parallel_numpy(qrels, run, k, rel_lvl):
    """NumPy fallback implementation."""
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in range(len(qrels)):
        scores[i] = _recall(qrels[i], run[i], k, rel_lvl)
    return scores


def _recall_parallel(qrels, run, k, rel_lvl):
    """Dispatch to best available implementation."""
    from ..config import use_numba

    if NUMBA_AVAILABLE and use_numba():
        return _recall_parallel_numba(qrels, run, k, rel_lvl)
    else:
        return _recall_parallel_numpy(qrels, run, k, rel_lvl)


# HIGH LEVEL FUNCTIONS =========================================================
def recall(
    qrels: Union[np.ndarray, list],
    run: Union[np.ndarray, list],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Recall (at k).

    **Recall** is the ratio between the retrieved documents that are relevant and the total number of relevant documents.<br />
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    $$
    \operatorname{Recall}=\frac{r}{R}
    $$

    where,

    - $r$ is the number of retrieved relevant documents;
    - $R$ is the total number of relevant documents.

    If k > 0,

    $$
    \operatorname{Recall@k}=\frac{r_k}{R}
    $$

    where,

    - $r_k$ is the number of retrieved relevant documents at k;
    - $R$ is the total number of relevant documents.

    Args:
        qrels: IDs and relevance scores of _relevant_ documents.

        run: IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Recall (at k) scores.
    """

    assert k >= 0, "k must be grater or equal to 0"

    return _recall_parallel(qrels, run, k, rel_lvl)
