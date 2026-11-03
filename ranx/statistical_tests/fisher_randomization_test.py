import numpy as np
from numba import njit, prange


@njit(cache=True)
def _swap_masks(n_permutations, n_queries, random_seed):
    # Drawn in a non-parallel function: Numba keeps one random state per thread and
    # np.random.seed only seeds the calling thread, so random draws inside a prange
    # loop are not reproducible (see issue #70).
    np.random.seed(random_seed)
    return np.random.random((n_permutations, n_queries)) < 0.5


@njit(cache=True)
def _abs_mean_diff(control, treatment, swap):
    a = 0.0
    b = 0.0
    for j in range(control.shape[0]):
        if swap[j]:
            a += treatment[j]
            b += control[j]
        else:
            a += control[j]
            b += treatment[j]
    n = control.shape[0]
    return abs(a / n - b / n)


@njit(cache=True, parallel=True)
def _count_at_least(control, treatment, masks, observed):
    counter_array = np.zeros(masks.shape[0])
    for i in prange(masks.shape[0]):
        if _abs_mean_diff(control, treatment, masks[i]) >= observed:
            counter_array[i] = 1.0
    return counter_array.mean()


def fisher_randomization_test(
    control, treatment, n_permutations=1000, max_p=0.01, random_seed=42
):
    """
    Performs (approximated) Fisher's Randomization Test.

    Null hypothesis: system A (control) and system B (treatment) are identical (i.e., system A has no effect compared to system B on the mean of a given performance metric)

    For further details, see Smucker et al. A Comparison of Statistical Significance Tests for Information Retrieval Evaluation, CIKM '07.

    Each permutation swaps the two systems' scores on every query independently with probability 0.5.
    The swaps are drawn up front from `random_seed`, so the result does not depend on the number of
    threads or on previous calls.

    """
    control = np.asarray(control, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    masks = _swap_masks(n_permutations, control.shape[0], random_seed)
    observed = _abs_mean_diff(
        control, treatment, np.zeros(control.shape[0], dtype=np.bool_)
    )
    p_value = _count_at_least(control, treatment, masks, observed)

    return p_value, p_value <= max_p
