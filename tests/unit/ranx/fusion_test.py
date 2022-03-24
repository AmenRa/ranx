import pytest
from ranx.run import Run
from ranx.fusion import weighted_sum, reciprocal_rank_fusion
import numpy as np
from numba.typed import List as TypedList

# FIXTURES =====================================================================
@pytest.fixture
def run_1():
    run_dict = {
        "q1": {
            "d1": 1,
            "d2": 2,
            "d3": 3,
        },
        "q2": {
            "d1": 1,
            "d2": 2,
        },
    }

    return Run(run_dict)


@pytest.fixture
def run_2():
    run_dict = {
        "q1": {
            "d1": 3,
            "d2": 2,
        },
        "q2": {
            "d1": 1,
            "d3": 3,
        },
    }

    return Run(run_dict)


@pytest.fixture
def run_3():
    run_dict = {
        "q1": {
            "d3": 1,
        },
        "q2": {
            "d2": 2,
            "d3": 3,
        },
    }

    return Run(run_dict)


# TESTS ========================================================================
def test_weighted_sum(run_1, run_2, run_3):
    weights = [0.3, 0.2, 0.5]
    combined_run = weighted_sum(
        TypedList([run_1.run, run_2.run, run_3.run]), np.array(weights)
    )

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


def test_reciprocal_rank(run_1, run_2, run_3):
    combined_run = reciprocal_rank_fusion(
        TypedList([run_1.run, run_2.run, run_3.run])
    )

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3
    
    assert combined_run["q1"]["d1"] == (1 / (60 + 3)) + (1 / (60 + 1))
    assert combined_run["q1"]["d2"] == (1 / (60 + 2)) + (1 / (60 + 2))
    assert combined_run["q1"]["d3"] == (1 / (60 + 1)) + (1 / (60 + 1))

    assert combined_run["q2"]["d1"] == (1 / (60 + 2)) + (1 / (60 + 2))
    assert combined_run["q2"]["d2"] == (1 / (60 + 1)) + (1 / (60 + 2))
    assert combined_run["q2"]["d3"] == (1 / (60 + 1)) + (1 / (60 + 1))
