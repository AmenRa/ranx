from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"

from .common import clean_qrels
from .hits import _hits


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _f1(qrels, run, k, rel_lvl):
    qrels = clean_qrels(qrels, rel_lvl)
    if len(qrels) == 0:
        return 0.0

    k = k if k != 0 else run.shape[0]
    if k == 0:
        return 0.0

    hits_score = _hits(qrels, run, k, rel_lvl)
    precision_score = hits_score / k
    recall_score = hits_score / qrels.shape[0]

    return 2 * (
        (precision_score * recall_score) / (precision_score + recall_score)
    )


@njit(cache=True, parallel=True)
def _f1_parallel(qrels, run, k, rel_lvl):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _f1(qrels[i], run[i], k, rel_lvl)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def f1(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute F1 (at k).

    **F1** is the harmonic mean of [**Precision**][ranx.metrics.precision] and [**Recall**][ranx.metrics.recall].<br />
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    $$
    \operatorname{F1} = 2 \times \frac{\operatorname{Precision} \times \operatorname{Recall}}{\operatorname{Precision} + \operatorname{Recall}}
    $$


    If k > 0,

    $$
    \operatorname{F1@k} = 2 \times \frac{\operatorname{Precision@k} \times \operatorname{Recall@k}}{\operatorname{Precision@k} + \operatorname{Recall@k}}
    $$

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        F1 (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _f1_parallel(qrels, run, k, rel_lvl)
