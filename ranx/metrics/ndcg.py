from typing import Union

import numpy as np

from ..decorators import maybe_njit
from .common import clean_qrels, fix_k


# LOW LEVEL FUNCTIONS ==========================================================
@maybe_njit(cache=True)
def _dcg(qrels, run, k, rel_lvl, jarvelin):
    qrels = clean_qrels(qrels, rel_lvl)
    if len(qrels) == 0:
        return 0.0

    k = fix_k(k, run)

    max_true_id = np.max(qrels[:, 0])
    min_true_id = np.min(qrels[:, 0])

    weighted_hit_list = np.zeros((k), dtype=np.float64)

    for i in range(k):
        if run[i, 0] > max_true_id:
            continue
        if run[i, 0] < min_true_id:
            continue
        for j in range(qrels.shape[0]):
            if run[i, 0] == qrels[j, 0]:
                weighted_hit_list[i] = qrels[j, 1]
                break

    if jarvelin:
        # Järvelin et al. formulation (see http://doi.acm.org/10.1145/582415.582418)
        return np.sum(weighted_hit_list / np.log2(np.arange(1, k + 1) + 1))

    else:
        # Burges et al. formulation (see https://doi.org/10.1145/1102351.1102363)
        return np.sum((2**weighted_hit_list - 1) / np.log2(np.arange(1, k + 1) + 1))


# Handle parallel version with conditional compilation
try:
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def _dcg_parallel_numba(qrels, run, k, rel_lvl, jarvelin):
        scores = np.zeros((len(qrels)), dtype=np.float64)
        for i in prange(len(qrels)):
            scores[i] = _dcg(qrels[i], run[i], k, rel_lvl, jarvelin)
        return scores

    @njit(cache=True, parallel=True)
    def _ndcg_parallel_numba(qrels, run, k, rel_lvl, jarvelin):
        scores = np.zeros((len(qrels)), dtype=np.float64)
        for i in prange(len(qrels)):
            scores[i] = _ndcg(qrels[i], run[i], k, rel_lvl, jarvelin)
        return scores

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _dcg_numpy(qrels, run, k, rel_lvl, jarvelin):
    """NumPy fallback implementation."""
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in range(len(qrels)):
        scores[i] = _dcg(qrels[i], run[i], k, rel_lvl, jarvelin)
    return scores


def _dcg_parallel(qrels, run, k, rel_lvl, jarvelin):
    """Dispatch to best available implementation."""
    from ..config import use_numba

    if NUMBA_AVAILABLE and use_numba():
        return _dcg_parallel_numba(qrels, run, k, rel_lvl, jarvelin)
    else:
        return _dcg_numpy(qrels, run, k, rel_lvl, jarvelin)


@maybe_njit(cache=True)
def _idcg(qrels, k, rel_lvl, jarvelin):
    return _dcg(qrels, qrels, k, rel_lvl, jarvelin)


@maybe_njit(cache=True)
def _ndcg(qrels, run, k, rel_lvl, jarvelin):
    dcg_score = _dcg(qrels, run, k, rel_lvl, jarvelin)
    idcg_score = _idcg(qrels, k, rel_lvl, jarvelin)

    # For numerical stability
    if idcg_score == 0.0:
        idcg_score = 1.0

    return dcg_score / idcg_score


def _ndcg_numpy(qrels, run, k, rel_lvl, jarvelin):
    """NumPy fallback implementation."""
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in range(len(qrels)):
        scores[i] = _ndcg(qrels[i], run[i], k, rel_lvl, jarvelin)
    return scores


def _ndcg_parallel(qrels, run, k, rel_lvl, jarvelin):
    """Dispatch to best available implementation."""
    from ..config import use_numba

    if NUMBA_AVAILABLE and use_numba():
        return _ndcg_parallel_numba(qrels, run, k, rel_lvl, jarvelin)
    else:
        return _ndcg_numpy(qrels, run, k, rel_lvl, jarvelin)


# HIGH LEVEL FUNCTIONS =========================================================
def dcg(
    qrels: Union[np.ndarray, list],
    run: Union[np.ndarray, list],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Discounted Cumulative Gain (DCG) as proposed by [Järvelin et al.](http://doi.acm.org/10.1145/582415.582418).<br />
    If k > 0, only the top-k retrieved documents are considered.

    $$
    \operatorname{DCG} = \frac{\operatorname{rel}_i}{\log_2(i+1)}
    $$

    where,

    - $\operatorname{rel}_i$ is the relevance value of the result at position i.

    ```bibtex
        @article{DBLP:journals/tois/JarvelinK02,
            author    = {Kalervo J{\"{a}}rvelin and
                        Jaana Kek{\"{a}}l{\"{a}}inen},
            title     = {Cumulated gain-based evaluation of {IR} techniques},
            journal   = {{ACM} Trans. Inf. Syst.},
            volume    = {20},
            number    = {4},
            pages     = {422--446},
            year      = {2002}
        }
    ```

    Args:
        qrels: IDs and relevance scores of _relevant_ documents.

        run: IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Discounted Cumulative Gain (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _dcg_parallel(qrels, run, k, rel_lvl, jarvelin=True)


def ndcg(
    qrels: Union[np.ndarray, list],
    run: Union[np.ndarray, list],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Normalized Discounted Cumulative Gain (NDCG) as proposed by [Järvelin et al.](http://doi.acm.org/10.1145/582415.582418).<br />
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    $$
    \operatorname{nDCG} = \frac{\operatorname{DCG}}{\operatorname{IDCG}}
    $$

    where,

    - $\operatorname{DCG}$ is Discounted Cumulative Gain;
    - $\operatorname{IDCG}$ is Ideal Discounted Cumulative Gain (max possible DCG).

    If k > 0,

    $$
    \operatorname{nDCG}_k = \frac{\operatorname{DCG}_k}{\operatorname{IDCG}_k}
    $$

    where,

    - $\operatorname{DCG}_k$ is Discounted Cumulative Gain at k;
    - $\operatorname{IDCG}_k$ is Ideal Discounted Cumulative Gain at k (max possible DCG at k).

    ```bibtex
        @article{DBLP:journals/tois/JarvelinK02,
            author    = {Kalervo J{\"{a}}rvelin and
                        Jaana Kek{\"{a}}l{\"{a}}inen},
            title     = {Cumulated gain-based evaluation of {IR} techniques},
            journal   = {{ACM} Trans. Inf. Syst.},
            volume    = {20},
            number    = {4},
            pages     = {422--446},
            year      = {2002}
        }
    ```

    Args:
        qrels: IDs and relevance scores of _relevant_ documents.

        run: IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Normalized Discounted Cumulative Gain (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _ndcg_parallel(qrels, run, k, rel_lvl, jarvelin=True)


def dcg_burges(
    qrels: Union[np.ndarray, list],
    run: Union[np.ndarray, list],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Discounted Cumulative Gain (DCG) at k as proposed by [Burges et al.](https://doi.org/10.1145/1102351.1102363).<br />
    If k > 0, only the top-k retrieved documents are considered.

    $$
    \operatorname{DCG} = \frac{2^{\operatorname{rel}_i-1}}{\log_2(i+1)}
    $$

    where,

    - $\operatorname{rel}_i$ is the relevance value of the result at position i.

    ```bibtex
        @inproceedings{DBLP:conf/icml/BurgesSRLDHH05,
            author    = {Christopher J. C. Burges and
                        Tal Shaked and
                        Erin Renshaw and
                        Ari Lazier and
                        Matt Deeds and
                        Nicole Hamilton and
                        Gregory N. Hullender},
            title     = {Learning to rank using gradient descent},
            booktitle = {{ICML}},
            series    = {{ACM} International Conference Proceeding Series},
            volume    = {119},
            pages     = {89--96},
            publisher = {{ACM}},
            year      = {2005}
        }
    ```

    Args:
        qrels: IDs and relevance scores of _relevant_ documents.

        run: IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Discounted Cumulative Gain (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _dcg_parallel(qrels, run, k, rel_lvl, jarvelin=False)


def ndcg_burges(
    qrels: Union[np.ndarray, list],
    run: Union[np.ndarray, list],
    k: int = 0,
    rel_lvl: int = 1,
) -> np.ndarray:
    r"""Compute Normalized Discounted Cumulative Gain (NDCG) at k as proposed by [Burges et al.](https://doi.org/10.1145/1102351.1102363).<br />
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    $$
    \operatorname{nDCG} = \frac{\operatorname{DCG}}{\operatorname{IDCG}}
    $$

    where,

    - $\operatorname{DCG}$ is Discounted Cumulative Gain;
    - $\operatorname{IDCG}$ is Ideal Discounted Cumulative Gain (max possible DCG).

    If k > 0,

    $$
    \operatorname{nDCG}_k = \frac{\operatorname{DCG}_k}{\operatorname{IDCG}_k}
    $$

    where,

    - $\operatorname{DCG}_k$ is Discounted Cumulative Gain at k;
    - $\operatorname{IDCG}_k$ is Ideal Discounted Cumulative Gain at k (max possible DCG at k).

    ```bibtex
        @inproceedings{DBLP:conf/icml/BurgesSRLDHH05,
            author    = {Christopher J. C. Burges and
                        Tal Shaked and
                        Erin Renshaw and
                        Ari Lazier and
                        Matt Deeds and
                        Nicole Hamilton and
                        Gregory N. Hullender},
            title     = {Learning to rank using gradient descent},
            booktitle = {{ICML}},
            series    = {{ACM} International Conference Proceeding Series},
            volume    = {119},
            pages     = {89--96},
            publisher = {{ACM}},
            year      = {2005}
        }
    ```

    Args:
        qrels: IDs and relevance scores of _relevant_ documents.

        run: IDs and relevance scores of _retrieved_ documents.

        k (int, optional): Number of retrieved documents to consider. k=0 means all retrieved documents will be considered. Defaults to 0.

        rel_lvl (int, optional): Minimum relevance judgment score to consider a document to be relevant. E.g., rel_lvl=1 means all documents with relevance judgment scores greater or equal to 1 will be considered relevant. Defaults to 1.

    Returns:
        Normalized Discounted Cumulative Gain (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _ndcg_parallel(qrels, run, k, rel_lvl, jarvelin=False)
