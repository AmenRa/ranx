import numpy as np
from numba import config, njit, prange

config.THREADING_LAYER = "workqueue"


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _clean_qrels(qrels):
    return qrels[np.nonzero(qrels[:, 1])]


@njit(cache=True)
def fix_k(k, run):
    if k == 0 or k > run.shape[0]:
        return run.shape[0]
    else:
        return k


@njit(cache=True)
def _hits(qrels, run, k):
    qrels = _clean_qrels(qrels.copy())
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


@njit(cache=True, parallel=True)
def _hits_parallel(qrels, run, k):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _hits(qrels[i], run[i], k)
    return scores


@njit(cache=True)
def _precision(qrels, run, k):
    qrels = _clean_qrels(qrels.copy())
    if len(qrels) == 0:
        return 0.0
    k = k if k != 0 else run.shape[0]
    return _hits(qrels, run, k) / k


@njit(cache=True, parallel=True)
def _precision_parallel(qrels, run, k):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _precision(qrels[i], run[i], k)
    return scores


@njit(cache=True)
def _recall(qrels, run, k):
    qrels = _clean_qrels(qrels.copy())
    if len(qrels) == 0:
        return 0.0
    k = k if k != 0 else run.shape[0]
    return _hits(qrels, run, k) / qrels.shape[0]


@njit(cache=True, parallel=True)
def _recall_parallel(qrels, run, k):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _recall(qrels[i], run[i], k)
    return scores


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


@njit(cache=True)
def _reciprocal_rank(qrels, run, k):
    qrels = _clean_qrels(qrels.copy())
    if len(qrels) == 0:
        return 0.0

    k = fix_k(k, run)

    for i in range(k):
        if run[i, 0] in qrels[:, 0]:
            return 1 / (i + 1)
    return 0.0


@njit(cache=True, parallel=True)
def _reciprocal_rank_parallel(qrels, run, k):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _reciprocal_rank(qrels[i], run[i], k)
    return scores


@njit(cache=True)
def _average_precision(qrels, run, k):
    qrels = _clean_qrels(qrels.copy())
    if len(qrels) == 0:
        return 0.0

    k = fix_k(k, run)

    hit_list = np.zeros((k), dtype=np.float64)

    max_true_id = np.max(qrels[:, 0])
    min_true_id = np.min(qrels[:, 0])

    for i in range(k):
        if run[i, 0] > max_true_id:
            continue
        if run[i, 0] < min_true_id:
            continue
        for j in range(qrels.shape[0]):
            if run[i, 0] == qrels[j, 0]:
                hit_list[i] = 1.0
                break

    precision_scores = np.zeros((k), dtype=np.float64)

    for r in range(k):
        if hit_list[r]:
            # Compute precision at k without computing hit list at k again
            # same as _precision(qrels, run, r + 1)
            precision_scores[r] = np.sum(hit_list[: r + 1]) / (r + 1)

    return np.sum(precision_scores) / qrels.shape[0]


@njit(cache=True, parallel=True)
def _average_precision_parallel(qrels, run, k):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _average_precision(qrels[i], run[i], k)
    return scores


@njit(cache=True)
def _dcg(qrels, run, k, jarvelin):
    qrels = _clean_qrels(qrels.copy())
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
        return np.sum((2 ** weighted_hit_list - 1) / np.log2(np.arange(1, k + 1) + 1))


@njit(cache=True)
def _idcg(qrels, k, jarvelin):
    return _dcg(qrels, qrels, k, jarvelin)


@njit(cache=True)
def _ndcg(qrels, run, k, jarvelin):
    dcg_score = _dcg(qrels, run, k, jarvelin)
    idcg_score = _idcg(qrels, k, jarvelin)

    # For numerical stability
    if idcg_score == 0.0:
        idcg_score = 1.0

    return dcg_score / idcg_score


@njit(cache=True, parallel=True)
def _ndcg_parallel(qrels, run, k, jarvelin):
    scores = np.zeros((len(qrels)), dtype=np.float64)
    for i in prange(len(qrels)):
        scores[i] = _ndcg(qrels[i], run[i], k, jarvelin)
    return scores


# HIGH LEVEL FUNCTIONS =========================================================
# BINARY METRICS ---------------------------------------------------------------
def hits(qrels, run, k=0):
    r"""Compute Hits (at k).

    *Hits* is the number of relevant documents retrieved.
    If k > 0, only the top-k retrieved documents are considered.

    Parameters
    ----------
    qrels : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    run : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of retrieved documents to consider. k=0 means all retrieved documents will be considered (default behaviour).

    Returns
    -------
    Float:
        Hits (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _hits_parallel(qrels, run, k)


def precision(qrels, run, k=0):
    r"""Compute Precision (at k).

    *Precision* is the proportion of the retrieved documents that are relevant.
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    .. math:: Precision={{r}\over{n}}

    where,

    - :math:`r` is the number of retrieved relevant documents;
    - :math:`n` is the number of retrieved documents.

    If k > 0,

    .. math:: Precision@k={{r}\over{k}}

    where,

    - :math:`r` is the number of retrieved relevant documents at k.


    Parameters
    ----------
    qrels : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    run : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of retrieved documents to consider. k=0 means all retrieved documents will be considered (default behaviour).

    Returns
    -------
    Float:
        Precision (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _precision_parallel(qrels, run, k)


def recall(qrels, run, k=0):
    r"""Compute Recall (at k).

    *Recall* is the ratio between the retrieved documents that are relevant and the total number of relevant documents.
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    .. math:: Recall={{r}\over{R}}

    where,

    - :math:`r` is the number of retrieved relevant documents;
    - :math:`R` is the total number of relevant documents.

    If k > 0,

    .. math:: Recall@k={{r}\over{R}}

    where,

    - :math:`r` is the number of retrieved relevant documents at k;
    - :math:`R` is the total number of relevant documents.

    Parameters
    ----------
    qrels : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    run : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of retrieved documents to consider. k=0 means all retrieved documents will be considered (default behaviour).

    Returns
    -------
    Float:
        Recall (at k) scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _recall_parallel(qrels, run, k)


def r_precision(qrels, run, k=0):
    r"""Compute R-precision.

    For a given query :math:`Q`, R-precision is the precision at :math:`R`, where :math:`R` is the number of relevant documents for :math:`Q`. In other words, if there are :math:`r` relevant documents among the top-:math:`R` retrieved documents, then R-precision is

    .. math:: R-Precision = \frac{r}{R}

    Parameters
    ----------
    qrels : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    run : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        This argument is ignored. It was added to standardize metrics' input.

    Returns
    -------
    FLoat
        R-precision scores.

    """

    return _r_precision_parallel(qrels, run)


def reciprocal_rank(qrels, run, k=0):
    r"""Compute Reciprocal Rank.

    The Reciprocal Rank is the multiplicative inverse of the rank of the first retrieved relevant document: 1 for first place, ​1/2 for second place, 1/3 for third place, and so on.
    If k > 0, only the top-k retrieved documents are considered.

    .. math:: Reciprocal Rank = \frac{1}{rank}

    where,

    - :math:`rank` is the position of the first retrieved relevant document.

    Parameters
    ----------
    qrels : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    run : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of retrieved documents to consider. k=0 means all retrieved documents will be considered (default behaviour).

    Returns
    -------
    Float
        Reciprocal Rank scores.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _reciprocal_rank_parallel(qrels, run, k)


def average_precision(qrels, run, k=0):
    r"""Compute Average Precision.

    Average Precision is the average of the Precision scores computed after each relevant document is retrieved.
    If k > 0, only the top-k retrieved documents are considered.

    .. math:: Average Precision = \frac{\sum_r Precision@r}{R}

    where,

    - :math:`r` is the position of a relevant document;
    - :math:`R` is the total number of relevant documents.

    Parameters
    ----------
    qrels : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    run : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of retrieved documents to consider. k=0 means all retrieved documents will be considered (default behaviour).

    Returns
    -------
    Float
        Mean Average Precision score.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _average_precision_parallel(qrels, run, k)


# NON-BINARY METRICS -----------------------------------------------------------
def ndcg(
    qrels,
    run,
    k=0,
):
    r"""Compute Normalized Discounted Cumulative Gain (NDCG) as proposed by Järvelin et al. (http://doi.acm.org/10.1145/582415.582418).
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    .. math:: nDCG = \frac{DCG}{IDCG}

    where,

    - :math:`DCG` is Discounted Cumulative Gain;
    - :math:`IDCG` is Ideal Discounted Cumulative Gain (max possibile DCG).

    If k > 0,

    .. math:: nDCG(k) = \frac{DCG(k)}{IDCG(k)}

    where,

    - :math:`DCG(k)` is Discounted Cumulative Gain at k;
    - :math:`IDCG(k)` is Ideal Discounted Cumulative Gain at k (max possibile DCG at k).

    .. code-block:: bibtex
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

    Parameters
    ----------
    qrels : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    run : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of retrieved documents to consider. k=0 means all retrieved documents will be considered (default behaviour).

    Returns
    -------
    Float
        Normalized Discounted Cumulative Gain at k score.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _ndcg_parallel(qrels, run, k, jarvelin=True)


def ndcg_burges(
    qrels,
    run,
    k=0,
):
    r"""Compute Normalized Discounted Cumulative Gain (NDCG) at k as proposed by Burges et al. (https://doi.org/10.1145/1102351.1102363).
    If k > 0, only the top-k retrieved documents are considered.

    If k = 0,

    .. math:: nDCG = \frac{DCG}{IDCG}

    where,

    - :math:`DCG` is Discounted Cumulative Gain;
    - :math:`IDCG` is Ideal Discounted Cumulative Gain (max possibile DCG).

    If k > 0,

    .. math:: nDCG(k) = \frac{DCG(k)}{IDCG(k)}

    where,

    - :math:`DCG(k)` is Discounted Cumulative Gain at k;
    - :math:`IDCG(k)` is Ideal Discounted Cumulative Gain at k (max possibile DCG at k).

    .. code-block:: bibtex

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

    Parameters
    ----------
    qrels : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    run : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of retrieved documents to consider. k=0 means all retrieved documents will be considered (default behaviour).

    Returns
    -------
    Float
        Normalized Discounted Cumulative Gain at k score.

    """

    assert k >= 0, "k must be grater or equal to 0"

    return _ndcg_parallel(qrels, run, k, jarvelin=False)
