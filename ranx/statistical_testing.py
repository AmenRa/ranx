import numpy as np
from numba import njit, prange
from scipy.stats import ttest_rel


@njit(cache=True)
def permute(x):
    permuted_x = np.empty_like(x)

    for i in prange(x.shape[0]):
        permuted_x[i] = np.random.permutation(x[i])

    return permuted_x


@njit(cache=True, parallel=True)
def fisher_randomization_test(
    control, treatment, n_permutations=1000, max_p=0.01, random_seed=42
):
    """
    Performs (approximated) Fisher's Randomization Test.

    Null hypotesis: system A (control) and system B (treatment) are identical (i.e., system A has no effect compared to system B on the mean of a given performance metric)

    For further details, see Smucker et al. A Comparison of Statistical Significance Tests for Information Retrieval Evaluation, CIKM '07.

    """
    np.random.seed(random_seed)

    control_mean = control.mean()
    treatment_mean = treatment.mean()
    control_treatment_diff = abs(control_mean - treatment_mean)
    control_treatment_stack = np.column_stack((control, treatment))

    counter_array = np.zeros(n_permutations)

    for i in prange(n_permutations):
        permuted = permute(control_treatment_stack)

        permuted_diff = abs(permuted[:, 0].mean() - permuted[:, 1].mean())

        if permuted_diff >= control_treatment_diff:
            counter_array[i] = 1.0

    p_value = counter_array.mean()

    return p_value, p_value <= max_p


def paired_student_t_test(control, treatment, max_p=0.01):
    """
    Two-sided Paired Student's t-Test.

    Null hypotesis: system A (control) and system B (treatment) are identical (i.e., system A has no effect compared to system B on the mean of a given performance metric)

    For further details, see https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples.

    """
    _, p_value = ttest_rel(control, treatment, alternative="two-sided")

    return p_value, p_value <= max_p
