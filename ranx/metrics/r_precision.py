from typing import Union

import numba
import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"

from .common import _clean_qrels
from .precision import _precision


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _r_precision(qrels, run):
    qrels = _clean_qrels(qrels.copy())
    if len(qrels) == 0:
        return 0.0
    return _precision(qrels, run, qrels.shape[0])


@njit(cache=True, parallel=True)
def _r_precision_parallel(qrels, run):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _r_precision(qrels[i], run[i])
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
def r_precision(
    qrels: Union[np.ndarray, numba.typed.List],
    run: Union[np.ndarray, numba.typed.List],
    k: int = 0,
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

    Returns:
        R-Precision scores.

    """

    return _r_precision_parallel(qrels, run)
