from math import isclose

import pytest
from ranx import Run
from ranx.normalization import borda_norm


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
def test_borda_norm(run_1, run_2, run_3):
    run_1_copy, run_2_copy, run_3_copy = (
        run_1.run.copy(),
        run_2.run.copy(),
        run_3.run.copy(),
    )

    norm_run_1, norm_run_2, norm_run_3 = borda_norm([run_1, run_2, run_3])

    assert run_1.run == run_1_copy
    assert run_2.run == run_2_copy
    assert run_3.run == run_3_copy

    assert (len(norm_run_1)) == 2
    assert len(norm_run_1["q1"]) == 3
    assert len(norm_run_1["q2"]) == 3

    assert (len(norm_run_2)) == 2
    assert len(norm_run_2["q1"]) == 3
    assert len(norm_run_2["q2"]) == 3

    assert (len(norm_run_3)) == 2
    assert len(norm_run_3["q1"]) == 3
    assert len(norm_run_3["q2"]) == 3

    assert isclose(norm_run_1["q1"]["d1"], 1 / 3)
    assert isclose(norm_run_1["q1"]["d2"], 2 / 3)
    assert isclose(norm_run_1["q1"]["d3"], 3 / 3)
    assert isclose(norm_run_1["q2"]["d1"], 2 / 3)
    assert isclose(norm_run_1["q2"]["d2"], 3 / 3)
    assert isclose(norm_run_1["q2"]["d3"], 1 / 3)

    assert isclose(norm_run_2["q1"]["d1"], 3 / 3)
    assert isclose(norm_run_2["q1"]["d2"], 2 / 3)
    assert isclose(norm_run_2["q1"]["d3"], 1 / 3)
    assert isclose(norm_run_2["q2"]["d1"], 2 / 3)
    assert isclose(norm_run_2["q2"]["d2"], 1 / 3)
    assert isclose(norm_run_2["q2"]["d3"], 3 / 3)

    assert isclose(norm_run_3["q1"]["d1"], 1.5 / 3)
    assert isclose(norm_run_3["q1"]["d2"], 1.5 / 3)
    assert isclose(norm_run_3["q1"]["d3"], 3 / 3)
    assert isclose(norm_run_3["q2"]["d1"], 1 / 3)
    assert isclose(norm_run_3["q2"]["d2"], 2 / 3)
    assert isclose(norm_run_3["q2"]["d3"], 3 / 3)
