import numba as nb
import numpy as np
from numba import njit, prange, set_num_threads


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=False, parallel=True)
def _parallelize(f, y_true, y_pred, k):
    scores = np.zeros((len(y_true)), dtype=np.float64)
    for i in prange(len(y_true)):
        scores[i] = f(y_true[i], y_pred[i], k)
    return scores


@njit(cache=False)
def _hits_at_k(y_true, y_pred, k):
    k = k if k <= y_pred.shape[0] else y_pred.shape[0]
    hits = 0.0

    max_true_id = np.max(y_true[:, 0])
    min_true_id = np.min(y_true[:, 0])

    for i in range(k):
        if y_pred[i, 0] > max_true_id:
            continue
        if y_pred[i, 0] < min_true_id:
            continue
        for j in range(y_true.shape[0]):
            if y_pred[i, 0] == y_true[j, 0]:
                hits += 1.0
                break

    return hits


@njit(cache=False)
def _hits_at_k_parallel(y_true, y_pred, k):
    return _parallelize(_hits_at_k, y_true, y_pred, k)


@njit(cache=False)
def _precision_at_k(y_true, y_pred, k):
    return _hits_at_k(y_true, y_pred, k) / k


@njit(cache=False)
def _precision_at_k_parallel(y_true, y_pred, k):
    return _parallelize(_precision_at_k, y_true, y_pred, k)


@njit(cache=False)
def _recall_at_k(y_true, y_pred, k):
    return _hits_at_k(y_true, y_pred, k) / y_true.shape[0]


@njit(cache=False)
def _recall_at_k_parallel(y_true, y_pred, k):
    return _parallelize(_recall_at_k, y_true, y_pred, k)


@njit(cache=False)
def _r_precision(y_true, y_pred):
    return _precision_at_k(y_true, y_pred, y_true.shape[0])


@njit(cache=False, parallel=True)
def _r_precision_parallel(y_true, y_pred):
    scores = np.zeros((len(y_true)), dtype=np.float64)
    for i in prange(len(y_true)):
        scores[i] = _r_precision(y_true[i], y_pred[i])
    return scores


@njit(cache=False)
def _reciprocal_rank(y_true, y_pred, k):
    k = k if k <= y_pred.shape[0] else y_pred.shape[0]
    for i in range(k):
        if y_pred[i, 0] in y_true[:, 0]:
            return 1 / (i + 1)
    return 0.0


@njit(cache=False)
def _mrr(y_true, y_pred, k):
    return _parallelize(_reciprocal_rank, y_true, y_pred, k)


@njit(cache=False)
def _average_precision(y_true, y_pred, k):
    k = k if k <= y_pred.shape[0] else y_pred.shape[0]

    hit_list = np.zeros((k), dtype=np.float64)

    max_true_id = np.max(y_true[:, 0])
    min_true_id = np.min(y_true[:, 0])

    for i in range(k):
        if y_pred[i, 0] > max_true_id:
            continue
        if y_pred[i, 0] < min_true_id:
            continue
        for j in range(y_true.shape[0]):
            if y_pred[i, 0] == y_true[j, 0]:
                hit_list[i] = 1.0
                break

    precision_scores = np.zeros((k), dtype=np.float64)

    for r in range(k):
        if hit_list[r]:
            # Compute precision at k without computing hit list at k again
            # same as _precision_at_k(y_true, y_pred, r + 1)
            precision_scores[r] = np.sum(hit_list[: r + 1]) / (r + 1)

    return np.sum(precision_scores) / y_true.shape[0]


@njit(cache=False)
def _map(y_true, y_pred, k):
    return _parallelize(_average_precision, y_true, y_pred, k)


@njit(cache=False)
def _dcg(y_true, y_pred, k, trec_eval):
    k = k if k <= y_pred.shape[0] else y_pred.shape[0]

    weighted_hit_list = np.zeros((k), dtype=np.float64)

    max_true_id = np.max(y_true[:, 0])
    min_true_id = np.min(y_true[:, 0])

    for i in range(k):
        if y_pred[i, 0] > max_true_id:
            continue
        if y_pred[i, 0] < min_true_id:
            continue
        for j in range(y_true.shape[0]):
            if y_pred[i, 0] == y_true[j, 0]:
                weighted_hit_list[i] = y_true[j, 1]
                break

    if trec_eval:
        return np.sum(weighted_hit_list / np.log2(np.arange(1, k + 1) + 1))

    else:
        # Standard formulation
        return np.sum((2 ** weighted_hit_list - 1) / np.log2(np.arange(1, k + 1) + 1))


@njit(cache=False)
def _idcg(y_true, k, trec_eval):
    return _dcg(y_true, y_true, k, trec_eval)


@njit(cache=False)
def _ndcg(y_true, y_pred, k, trec_eval):
    dcg_score = _dcg(y_true, y_pred, k, trec_eval)
    idcg_score = _idcg(y_true, k, trec_eval)

    return dcg_score / idcg_score


@njit(cache=False, parallel=True)
def _ndcg_parallel(y_true, y_pred, k, trec_eval):
    scores = np.zeros((len(y_true)), dtype=np.float64)
    for i in prange(len(y_true)):
        scores[i] = _ndcg(y_true[i], y_pred[i], k, trec_eval)
    return scores


@njit(cache=False)
def _descending_sort(x):
    return x[np.argsort(x[:, 1])[::-1]]


@njit(cache=False, parallel=True)
def _descending_sort_parallel(x):
    for i in prange(len(x)):
        x[i] = _descending_sort(x[i])
    return x


# HIGH LEVEL FUNCTIONS =========================================================
def _choose_optimal_function(
    y_true,
    y_pred,
    f_name,
    f_single,
    f_parallel,
    f_additional_args={},
    return_mean=True,
    sort=False,
    threads=1,
):
    set_num_threads(threads)

    # Check y_true -------------------------------------------------------------
    if type(y_true) == np.ndarray and y_true.ndim == 2:
        y_true_type = "single"
    elif type(y_true) == np.ndarray and y_true.ndim == 3:
        y_true_type = "parallel"
    elif type(y_true) == nb.typed.typedlist.List and y_true[0].ndim == 1:
        y_true_type = "single"
    elif type(y_true) == nb.typed.typedlist.List and y_true[0].ndim == 2:
        y_true_type = "parallel"
    else:
        raise TypeError("y_true type not supported.")

    # Check y_pred -------------------------------------------------------------
    if type(y_pred) == np.ndarray and y_pred.ndim == 2:
        y_pred_type = "single"
    elif type(y_pred) == np.ndarray and y_pred.ndim == 3:
        y_pred_type = "parallel"
    elif type(y_pred) == nb.typed.typedlist.List and y_pred[0].ndim == 1:
        y_pred_type = "single"
    elif type(y_pred) == nb.typed.typedlist.List and y_pred[0].ndim == 2:
        y_pred_type = "parallel"
    else:
        raise TypeError("y_pred type not supported.")

    # Check y_true and y_pred are aligned --------------------------------------
    if y_true_type != y_pred_type:
        raise TypeError("y_true and y_pred types not supported.")

    if y_true_type == "single":
        if f_name == "ndcg":
            y_true = _descending_sort(y_true)
        if sort:
            y_pred = _descending_sort(y_pred)
        return f_single(y_true, y_pred, **f_additional_args)
    else:
        if f_name == "ndcg":
            y_true = _descending_sort_parallel(y_true)
        if sort:
            y_pred = _descending_sort_parallel(y_pred)
        scores = f_parallel(y_true, y_pred, **f_additional_args)
        if return_mean:
            return np.mean(scores)
        return scores


# BINARY METRICS ---------------------------------------------------------------
def hits_at_k(y_true, y_pred, k, return_mean=True, sort=False, threads=1):
    r"""Compute Hits at k.

    *Hits at k* is the number of relevant documents in the top-k retrieved documents.

    Parameters
    ----------
    y_true : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    y_pred : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of results to consider.
    
    return_mean : Bool
        Whether to return the average of the scores or a list scores (one per query).
        Default is `True`.

    sort : Bool
        Whether to sort y_true and y_pred by descending relevance scores.
        Default is `False`.

    threads : Int
        Number of threads to use for computation.

    Returns
    -------
    Float:
        Hits at k score.

    """

    return _choose_optimal_function(
        y_true=y_true,
        y_pred=y_pred,
        f_name="hits_at_k",
        f_single=_hits_at_k,
        f_parallel=_hits_at_k_parallel,
        f_additional_args={"k": k},
        return_mean=return_mean,
        sort=sort,
        threads=threads,
    )


def precision_at_k(y_true, y_pred, k, return_mean=True, sort=False, threads=1):
    r"""Compute Precision at k.

    *Precision at k* (P@k) is the proportion of the top-k retrieved documents that are relevant.

    .. math:: P@k={{r}\over{k}}

    where,

    - :math:`r` is the number of relevant documents.

    Parameters
    ----------
    y_true : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    y_pred : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of results to consider.
    
    return_mean : Bool
        Whether to return the average of the scores or a list scores (one per query).
        Default is `True`.

    sort : Bool
        Whether to sort y_true and y_pred by descending relevance scores.
        Default is `False`.

    threads : Int
        Number of threads to use for computation.

    Returns
    -------
    Float:
        Precision at k score.

    """

    return _choose_optimal_function(
        y_true=y_true,
        y_pred=y_pred,
        f_name="precision_at_k",
        f_single=_precision_at_k,
        f_parallel=_precision_at_k_parallel,
        f_additional_args={"k": k},
        return_mean=return_mean,
        sort=sort,
        threads=threads,
    )


def recall_at_k(y_true, y_pred, k, return_mean=True, sort=False, threads=1):
    r"""Compute Recall at k.

    *Recall at k* (P@k) is the ratio between the top-k retrieved documents that are relevant and the total number of relevant documents.

    .. math:: R@k={{r}\over{R}}

    where,

    - :math:`r` is the number of retrieved relevant documents at k.
    - :math:`R` is the total number of relevant documents.

    Parameters
    ----------
    y_true : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    y_pred : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of results to consider.
    
    return_mean : Bool
        Whether to return the average of the scores or a list scores (one per query).
        Default is `True`.

    sort : Bool
        Whether to sort y_true and y_pred by descending relevance scores.
        Default is `False`.

    threads : Int
        Number of threads to use for computation.

    Returns
    -------
    Float:
        Recall at k score.

    """

    return _choose_optimal_function(
        y_true=y_true,
        y_pred=y_pred,
        f_name="recall_at_k",
        f_single=_recall_at_k,
        f_parallel=_recall_at_k_parallel,
        f_additional_args={"k": k},
        return_mean=return_mean,
        sort=sort,
        threads=threads,
    )


def r_precision(y_true, y_pred, return_mean=True, sort=False, threads=1):
    r"""Compute R-precision.

    For a given query topic :math:`Q`, R-precision is the precision at :math:`R`, where :math:`R` is the number of relevant documents for :math:`Q`. In other words, if there are :math:`r` relevant documents among the top-:math:`R` retrieved documents, then R-precision is

    .. math:: \frac{r}{R}

    Parameters
    ----------
    y_true : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    y_pred : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.
    
    return_mean : Bool
        Whether to return the average of the scores or a list scores (one per query).
        Default is `True`.

    sort : Bool
        Whether to sort y_true and y_pred by descending relevance scores.
        Default is `False`.

    threads : Int
        Number of threads to use for computation.

    Returns
    -------
    FLoat
        R-precision score.

    """

    return _choose_optimal_function(
        y_true=y_true,
        y_pred=y_pred,
        f_name="r_precision",
        f_single=_r_precision,
        f_parallel=_r_precision_parallel,
        f_additional_args={},
        return_mean=return_mean,
        sort=sort,
        threads=threads,
    )


def mrr(y_true, y_pred, k, return_mean=True, sort=False, threads=1):
    r"""Compute Mean Reciprocal Rank.

    The Mean Reciprocal Rank is a statistic measure for evaluating any process that produces a list of possible responses to a sample of queries, ordered by probability of correctness. The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer: 1 for first place, ​1/2 for second place, 1/3 for third place and so on. The mean reciprocal rank is the average of the reciprocal ranks of results for a sample of queries.

    .. math:: MRR = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}

    where,

    - :math:`N` is the number of ranked lists;
    - :math:`rank_i` is the position of the correct document for the ranked list :math:`i`.

    Parameters
    ----------
    y_true : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    y_pred : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of results to consider.
    
    return_mean : Bool
        Whether to return the average of the scores or a list scores (one per query).
        Default is `True`.

    sort : Bool
        Whether to sort y_true and y_pred by descending relevance scores.
        Default is `False`.

    threads : Int
        Number of threads to use for computation.

    Returns
    -------
    Float
        Mean Reciprocal Rank score.

    """

    return _choose_optimal_function(
        y_true=y_true,
        y_pred=y_pred,
        f_name="mrr",
        f_single=_reciprocal_rank,
        f_parallel=_mrr,
        f_additional_args={"k": k},
        return_mean=return_mean,
        sort=sort,
        threads=threads,
    )


def map(y_true, y_pred, k, return_mean=True, sort=False, threads=1):
    r"""Compute Mean Average Precision.

    The Mean Average Precision (MAP) is the arithmetic mean of the Average Precision scores of a set of ranked lists.

    .. math:: MAP = {1\over n}\sum\limits_n {AP_n }

    where,

    - :math:`n` is the number of ranked lists;
    - :math:`AP_n` is the :math:`Average\,Precision` of :math:`n`-th ranked list.

    Parameters
    ----------
    y_true : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    y_pred : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of results to consider.
    
    return_mean : Bool
        Whether to return the average of the scores or a list scores (one per query).
        Default is `True`.

    sort : Bool
        Whether to sort y_true and y_pred by descending relevance scores.
        Default is `False`.

    threads : Int
        Number of threads to use for computation.

    Returns
    -------
    Float
        Mean Average Precision score.

    """

    return _choose_optimal_function(
        y_true=y_true,
        y_pred=y_pred,
        f_name="average_precision",
        f_single=_average_precision,
        f_parallel=_map,
        f_additional_args={"k": k},
        return_mean=return_mean,
        sort=sort,
        threads=threads,
    )


# NON-BINARY METRICS -----------------------------------------------------------
def ndcg(
    y_true, y_pred, k, return_mean=True, sort=False, trec_eval=False, threads=1,
):
    r"""Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    .. math:: nDCG(k) = \frac{DCG(k)}{IDCG(k)}

    where,

    - :math:`DCG(k)` is Discounted Cumulative Gain at k;
    - :math:`IDCG(k)` is Ideal Discounted Cumulative Gain at k (max possibile DCG at k).

    .. code-block:: bibtex

        @article{jarvelin2002cumulated,
            title="Cumulated gain-based evaluation of IR techniques",
            author="J{\"a}rvelin, Kalervo and Kek{\"a}l{\"a}inen, Jaana",
            journal="ACM Transactions on Information Systems (TOIS)",
            volume="20",
            number="4",
            pages="422--446",
            year="2002",
            publisher="ACM",
            doi="10.1145/582415.582418",
        }

    Example:

    .. code-block:: python
    
        >>> y_true = np.array([[[12, 0.5], [25, 0.3]], [[11, 0.4], [2, 0.6]]])
        >>> y_pred = np.array([[[12, 0.9], [234, 0.8], [25, 0.7], [36, 0.6], [32, 0.5], [35, 0.4]], [[12, 0.9], [11, 0.8], [25, 0.7], [36, 0.6], [2, 0.5], [35, 0.4]]])
        >>> k = 5
        >>> ndcg(y_true, y_pred, k)
        0.7525653965843032

    Parameters
    ----------
    y_true : Numpy Array or Numba Typed List
        IDs and relevance scores of _relevant_ documents.

    y_pred : Numpy Array or Numba Typed List
        IDs and relevance scores of _retrieved_ documents.

    k : Int
        Number of results to consider.
    
    return_mean : Bool
        Whether to return the average of the scores or a list scores (one per query).
        Default is `True`.

    sort : Bool
        Whether to sort y_true and y_pred by descending relevance scores.
        Default is `False`.

    trec_eval : Bool
        `TREC Eval` uses a non-standard NDCG implementation. To mimic its behaviour, set this parameter to `True`.
        Default is `False`.

    threads : Int
        Number of threads to use for computation.

    Returns
    -------
    Float
        Normalized Discounted Cumulative Gain at k score.

    """

    return _choose_optimal_function(
        y_true=y_true,
        y_pred=y_pred,
        f_name="ndcg",
        f_single=_ndcg,
        f_parallel=_ndcg_parallel,
        f_additional_args={"k": k, "trec_eval": trec_eval},
        return_mean=return_mean,
        sort=sort,
        threads=threads,
    )
