from scipy.stats import ttest_rel


def paired_student_t_test(control, treatment, max_p=0.01):
    """
    Two-sided Paired Student's t-Test.

    Null hypotesis: system A (control) and system B (treatment) are identical (i.e., system A has no effect compared to system B on the mean of a given performance metric)

    For further details, see https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples.

    """
    _, p_value = ttest_rel(control, treatment, alternative="two-sided")

    return p_value, p_value <= max_p
