import numba
import numpy as np
import pytest

from ranx.statistical_tests import fisher_randomization_test


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    control = rng.random(500)
    treatment = np.clip(control + rng.normal(0.01, 0.1, 500), 0, 1)
    return control, treatment


def test_same_seed_gives_same_p_value_across_calls(scores):
    control, treatment = scores
    first = fisher_randomization_test(control, treatment, 1000, 0.05, 0)
    for _ in range(5):
        assert fisher_randomization_test(control, treatment, 1000, 0.05, 0) == first


def test_p_value_does_not_depend_on_thread_count(scores):
    control, treatment = scores
    results = set()
    for threads in sorted({1, numba.config.NUMBA_NUM_THREADS}):
        numba.set_num_threads(threads)
        results.add(fisher_randomization_test(control, treatment, 1000, 0.05, 0)[0])
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    assert len(results) == 1


def test_identical_systems_are_not_significant(scores):
    control, _ = scores
    p_value, significant = fisher_randomization_test(
        control, control.copy(), 200, 0.05, 1
    )
    assert p_value == 1.0 and not significant


def test_large_difference_is_significant():
    control = np.zeros(100)
    treatment = np.ones(100)
    p_value, significant = fisher_randomization_test(control, treatment, 500, 0.01, 3)
    # only the identity-like permutations reach the observed difference
    assert p_value < 0.01 and significant
