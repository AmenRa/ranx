import numpy as np
import pytest
from ranx.fusion import logn_isr
from ranx import Run


# FIXTURES =====================================================================
@pytest.fixture
def run_1():
    run_dict = {
        "q1": {"d1": 1, "d2": 2, "d3": 3},
        "q2": {"d1": 1, "d2": 2},
    }

    return Run(run_dict)


@pytest.fixture
def run_2():
    run_dict = {
        "q1": {"d1": 3, "d2": 2},
        "q2": {"d1": 1, "d3": 3},
    }

    return Run(run_dict)


@pytest.fixture
def run_3():
    run_dict = {
        "q1": {"d3": 1},
        "q2": {"d2": 2, "d3": 3},
    }

    return Run(run_dict)


# TESTS ========================================================================
def test_logn_isr(run_1, run_2, run_3):
    sigma = 0.1

    combined_run = logn_isr([run_1, run_2, run_3], sigma=sigma)

    assert combined_run.name == "logn_isr"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == (
        (1 / (3 ** 2)) + (1 / (1 ** 2))
    ) * np.log(2 + sigma)
    assert combined_run["q1"]["d2"] == (
        (1 / (2 ** 2)) + (1 / (2 ** 2))
    ) * np.log(2 + sigma)
    assert combined_run["q1"]["d3"] == (
        (1 / (1 ** 2)) + (1 / (1 ** 2))
    ) * np.log(2 + sigma)
    assert combined_run["q2"]["d1"] == (
        (1 / (2 ** 2)) + (1 / (2 ** 2))
    ) * np.log(2 + sigma)
    assert combined_run["q2"]["d2"] == (
        (1 / (1 ** 2)) + (1 / (2 ** 2))
    ) * np.log(2 + sigma)
    assert combined_run["q2"]["d3"] == (
        (1 / (1 ** 2)) + (1 / (1 ** 2))
    ) * np.log(2 + sigma)
