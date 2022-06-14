import pytest
from ranx import Run
from ranx.fusion import rbc


# FIXTURES =====================================================================
@pytest.fixture
def run_1():
    run_dict = {"q1": {"A": 7, "D": 6, "B": 5, "C": 4, "G": 3, "F": 2}}

    return Run(run_dict)


@pytest.fixture
def run_2():
    run_dict = {"q1": {"B": 7, "D": 6, "E": 5, "C": 4}}

    return Run(run_dict)


@pytest.fixture
def run_3():
    run_dict = {"q1": {"A": 7, "B": 6, "D": 5, "C": 4, "G": 3, "F": 2, "E": 1}}

    return Run(run_dict)


@pytest.fixture
def run_4():
    run_dict = {"q1": {"G": 7, "D": 6, "E": 5, "A": 4, "F": 3, "C": 2}}

    return Run(run_dict)


# TESTS ========================================================================
def test_rbc(run_1, run_2, run_3, run_4):
    combined_run = rbc([run_1, run_2, run_3, run_4], phi=0.6)

    assert combined_run.name == "rbc"

    assert len(combined_run) == 1
    assert len(combined_run["q1"]) == 7

    assert round(combined_run["q1"]["A"], 2) == 0.89
    assert round(combined_run["q1"]["D"], 2) == 0.86
    assert round(combined_run["q1"]["B"], 2) == 0.78
    assert round(combined_run["q1"]["G"], 2) == 0.50
    assert round(combined_run["q1"]["E"], 2) == 0.31
    assert round(combined_run["q1"]["C"], 2) == 0.29
    assert round(combined_run["q1"]["F"], 2) == 0.11

    combined_run = rbc([run_1, run_2, run_3, run_4], phi=0.8)

    assert combined_run.name == "rbc"

    assert len(combined_run) == 1
    assert len(combined_run["q1"]) == 7

    assert round(combined_run["q1"]["D"], 2) == 0.61
    assert round(combined_run["q1"]["A"], 2) == 0.50
    assert round(combined_run["q1"]["B"], 2) == 0.49
    assert round(combined_run["q1"]["C"], 2) == 0.37
    assert round(combined_run["q1"]["G"], 2) == 0.36
    assert round(combined_run["q1"]["E"], 2) == 0.31
    assert round(combined_run["q1"]["F"], 2) == 0.21

    combined_run = rbc([run_1, run_2, run_3, run_4], phi=0.9)

    assert combined_run.name == "rbc"

    assert len(combined_run) == 1
    assert len(combined_run["q1"]) == 7

    assert round(combined_run["q1"]["D"], 2) == 0.35
    assert round(combined_run["q1"]["C"], 2) == 0.28
    assert round(combined_run["q1"]["A"], 2) == 0.27
    assert round(combined_run["q1"]["B"], 2) == 0.27
    assert round(combined_run["q1"]["G"], 2) == 0.23
    assert round(combined_run["q1"]["E"], 2) == 0.22
    assert round(combined_run["q1"]["F"], 2) == 0.18

