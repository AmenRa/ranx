import numpy as np
import pytest
from numba.typed import List as TypedList
from ranx import Run
from ranx.fusion import condorcet
from ranx.fusion.condorcet import get_candidates, get_results


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
        "q1": {"d1": 2, "d2": 3},
        "q2": {},
    }

    return Run(run_dict)


@pytest.fixture
def run_6():
    run_dict = {
        "q1": {"d3": 3},
        "q2": {},
    }

    return Run(run_dict)


# TESTS ========================================================================
def test_get_candidates(run_1, run_2, run_3):
    candidates = get_candidates(TypedList([run_1.run, run_2.run, run_3.run]))

    assert set(candidates[0]) == {"d1", "d2", "d3"}
    assert set(candidates[1]) == {"d1", "d2", "d3"}


def test_get_results(run_1, run_2, run_3):
    results = get_results(TypedList([run_1.run, run_2.run, run_3.run]))

    assert results[0][0] == TypedList(["d3", "d2", "d1"])
    assert results[0][1] == TypedList(["d2", "d1"])
    assert results[0][2] == TypedList(["d3"])
    assert results[1][0] == TypedList(["d2", "d1"])
    assert results[1][1] == TypedList(["d3", "d1"])
    assert results[1][2] == TypedList(["d3", "d2"])


def test_condorcet(run_1, run_2, run_3):
    combined_run = condorcet([run_1, run_2, run_3])

    assert combined_run.name == "condorcet"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == 1
    assert combined_run["q1"]["d2"] == 2
    assert combined_run["q1"]["d3"] == 3
    assert combined_run["q2"]["d1"] == 1
    assert combined_run["q2"]["d2"] == 2
    assert combined_run["q2"]["d3"] == 3


def test_condorcet_no_result_query(run_4, run_5, run_6):
    combined_run = condorcet([run_4, run_5, run_6])

    assert combined_run.name == "condorcet"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 0

    assert combined_run["q1"]["d1"] == 1
    assert combined_run["q1"]["d2"] == 2
    assert combined_run["q1"]["d3"] == 3
