import pytest
from ranx import Run
from ranx.fusion import bordafuse


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


@pytest.fixture
def run_4():
    run_dict = {
        "q1": {"d1": 1, "d2": 2, "d3": 3},
        "q2": {},
    }

    return Run(run_dict)


@pytest.fixture
def run_5():
    run_dict = {
        "q1": {"d1": 3, "d2": 2},
        "q2": {},
    }

    return Run(run_dict)


@pytest.fixture
def run_6():
    run_dict = {
        "q1": {"d3": 1},
        "q2": {},
    }

    return Run(run_dict)


# TESTS ========================================================================
def test_bordafuse(run_1, run_2, run_3):
    combined_run = bordafuse([run_1, run_2, run_3])

    assert combined_run.name == "bordafuse"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == 1 + 3 + 1.5
    assert combined_run["q1"]["d2"] == 2 + 2 + 1.5
    assert combined_run["q1"]["d3"] == 3 + 1 + 3
    assert combined_run["q2"]["d1"] == 2 + 2 + 1
    assert combined_run["q2"]["d2"] == 3 + 1 + 2
    assert combined_run["q2"]["d3"] == 1 + 3 + 3


def test_bordafuse_no_result_query(run_4, run_5, run_6):
    combined_run = bordafuse([run_4, run_5, run_6])

    assert combined_run.name == "bordafuse"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 0

    assert combined_run["q1"]["d1"] == 1 + 3 + 1.5
    assert combined_run["q1"]["d2"] == 2 + 2 + 1.5
    assert combined_run["q1"]["d3"] == 3 + 1 + 3
