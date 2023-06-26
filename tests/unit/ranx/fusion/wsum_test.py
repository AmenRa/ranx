import pytest

from ranx import Run
from ranx.fusion import wsum


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
def test_wsum(run_1, run_2, run_3):
    weights = [0.3, 0.2, 0.5]
    combined_run = wsum([run_1, run_2, run_3], weights)

    assert combined_run.name == "weighted_sum"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert (
        combined_run["q1"]["d1"]
        == weights[0] * run_1["q1"]["d1"] + weights[1] * run_2["q1"]["d1"]
    )

    assert (
        combined_run["q1"]["d2"]
        == weights[0] * run_1["q1"]["d2"] + weights[1] * run_2["q1"]["d2"]
    )

    assert (
        combined_run["q1"]["d3"]
        == weights[0] * run_1["q1"]["d3"] + weights[2] * run_3["q1"]["d3"]
    )

    assert (
        combined_run["q2"]["d1"]
        == weights[0] * run_1["q2"]["d1"] + weights[1] * run_2["q2"]["d1"]
    )

    assert (
        combined_run["q2"]["d2"]
        == weights[0] * run_1["q2"]["d2"] + weights[2] * run_3["q2"]["d2"]
    )

    assert (
        combined_run["q2"]["d3"]
        == weights[1] * run_2["q2"]["d3"] + weights[2] * run_3["q2"]["d3"]
    )
