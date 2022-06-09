import pytest
from ranx.fusion import comb_gmnz, comb_mnz, comb_sum
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
def test_comb_gmnz(run_1, run_2, run_3):
    combined_run = comb_gmnz([run_1, run_2, run_3], 1.0)

    assert combined_run.name == "comb_gmnz"
    assert combined_run.run == comb_mnz([run_1, run_2, run_3]).run

    combined_run = comb_gmnz([run_1, run_2, run_3], 0.0)

    assert combined_run.name == "comb_gmnz"
    assert combined_run.run == comb_sum([run_1, run_2, run_3]).run
