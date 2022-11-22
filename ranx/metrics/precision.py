from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"

from .common import clean_qrels
from .hits import _hits


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _precision(qrels, run, k, rel_lvl):
    k = k if k != 0 else run.shape[0]
    if k == 0:
        return 0.0

    return _hits(qrels, run, k, rel_lvl) / k


@njit(cache=True, parallel=True)
def _precision_parallel(qrels, run, k, rel_lvl):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _precision(qrels[i], run[i], k, rel_lvl)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def precision(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Precision (at k).

    **Precision** is the proportion of the retrieved documents that are relevant.<br />
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    $$
    \operatorname{Precision}=\frac{r}{n}
    $$

    where,

    - $r$ is the number of retrieved relevant documents;
    - $n$ is the number of retrieved documents.

    If k > 0,

    $$
    \operatorname{Precision@k}=\frac{r_k}{k}
    $$

    where,

    - $r_k$ is the number of retrieved relevant documents at k.

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Precision (at k) scores.
    """

    assert k >= 0, "k must be grater or equal to 0"

    return _precision_parallel(qrels, run, k, rel_lvl)
