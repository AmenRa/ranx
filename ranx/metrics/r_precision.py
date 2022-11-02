from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"

from .common import clean_qrels
from .precision import _precision


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _r_precision(qrels, run, rel_lvl):
    qrels = clean_qrels(qrels, rel_lvl)

    if len(qrels) == 0:
        return 0.0

    return _precision(qrels, run, qrels.shape[0], rel_lvl)


@njit(cache=True, parallel=True)
def _r_precision_parallel(qrels, run, rel_lvl):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _r_precision(qrels[i], run[i], rel_lvl)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def r_precision(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute R-Precision.

    For a given query $Q$, R-Precision is the precision at $R$, where $R$ is the number of relevant documents for $Q$. In other words, if there are $r$ relevant documents among the top-$R$ retrieved documents, then R-precision is:

    $$
    \operatorname{R-Precision} = \frac{r}{R}
    $$

    Args:
        qrels (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _relevant_ documents.

        run (Union[np.ndarray, numba.typed.List]): IDs and relevance scores of _retrieved_ documents.

        k (int, optional): This argument is ignored. It was added to standardize metrics' input. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        R-Precision scores.

    """

    return _r_precision_parallel(qrels, run, rel_lvl)
