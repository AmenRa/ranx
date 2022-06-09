import pytest
from ranx import Run
from ranx.fusion import wmnz


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
def test_wmnz(run_1, run_2, run_3):
    combined_run = wmnz([run_1, run_2, run_3], [1, 1, 1])

    assert combined_run.name == "wmnz"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == sum([1, 3]) * 2
    assert combined_run["q1"]["d2"] == sum([2, 2]) * 2
    assert combined_run["q1"]["d3"] == sum([3, 1]) * 2
    assert combined_run["q2"]["d1"] == sum([1, 1]) * 2
    assert combined_run["q2"]["d2"] == sum([2, 2]) * 2
    assert combined_run["q2"]["d3"] == sum([3, 3]) * 2

    weights = [0.3, 0.2, 0.5]
    combined_run = wmnz([run_1, run_2, run_3], weights)

    assert combined_run.name == "wmnz"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == (
        run_1["q1"]["d1"] + run_2["q1"]["d1"]
    ) * (weights[0] + weights[1])

    assert combined_run["q1"]["d2"] == (
        run_1["q1"]["d2"] + run_2["q1"]["d2"]
    ) * (weights[0] + weights[1])

    assert combined_run["q1"]["d3"] == (
        run_1["q1"]["d3"] + run_3["q1"]["d3"]
    ) * (weights[0] + weights[2])

    assert combined_run["q2"]["d1"] == (
        run_1["q2"]["d1"] + run_2["q2"]["d1"]
    ) * (weights[0] + weights[1])

    assert combined_run["q2"]["d2"] == (
        run_1["q2"]["d2"] + run_3["q2"]["d2"]
    ) * (weights[0] + weights[2])

    assert combined_run["q2"]["d3"] == (
        run_2["q2"]["d3"] + run_3["q2"]["d3"]
    ) * (weights[1] + weights[2])

