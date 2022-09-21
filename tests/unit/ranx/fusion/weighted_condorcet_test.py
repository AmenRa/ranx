import numpy as np
import pytest
from ranx import Run
from ranx.fusion import weighted_condorcet


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
        "q1": {"d1": 2, "d2": 3},
        "q2": {"d1": 1, "d3": 3},
    }

    return Run(run_dict)


@pytest.fixture
def run_3():
    run_dict = {
        "q1": {"d3": 3},
        "q2": {"d2": 2, "d3": 3},
    }

    return Run(run_dict)


# TESTS ========================================================================
def test_weighted_condorcet(run_1, run_2, run_3):
    combined_run = weighted_condorcet([run_1, run_2, run_3], weights=[1, 1, 1])

    assert combined_run.name == "weighted_condorcet"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == 1
    assert combined_run["q1"]["d2"] == 2
    assert combined_run["q1"]["d3"] == 3
    assert combined_run["q2"]["d1"] == 1
    assert combined_run["q2"]["d2"] == 2
    assert combined_run["q2"]["d3"] == 3

    combined_run = weighted_condorcet([run_1, run_2, run_3], weights=[1, 5, 2])

    assert combined_run.name == "weighted_condorcet"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == 2
    assert combined_run["q1"]["d2"] == 3
    assert combined_run["q1"]["d3"] == 1
    assert combined_run["q2"]["d1"] == 2
    assert combined_run["q2"]["d2"] == 1
    assert combined_run["q2"]["d3"] == 3
