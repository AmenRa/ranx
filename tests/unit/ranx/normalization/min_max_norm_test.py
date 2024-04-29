import pytest

from ranx import Run
from ranx.normalization import min_max_norm, norm_switch


# FIXTURES =====================================================================
@pytest.fixture
def run():
    run_dict = {
        "q1": {"d1": 1, "d2": 2, "d3": 3},
        "q2": {"d1": 1, "d2": 2},
    }

    return Run(run_dict)


# TESTS ========================================================================
def test_min_max_norm(run):
    run_copy = run.run.copy()

    norm_run = min_max_norm(run)

    assert run.run == run_copy

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1
    assert run.run["q1"]["d2"] == 2
    assert run.run["q1"]["d3"] == 3
    assert run.run["q2"]["d1"] == 1
    assert run.run["q2"]["d2"] == 2
    assert run.size == 2

    assert len(norm_run) == 2
    assert len(norm_run["q1"]) == 3
    assert len(norm_run["q2"]) == 2
    assert norm_run["q1"]["d1"] == (1 - 1) / (3 - 1)
    assert norm_run["q1"]["d2"] == (2 - 1) / (3 - 1)
    assert norm_run["q1"]["d3"] == (3 - 1) / (3 - 1)
    assert norm_run["q2"]["d1"] == (1 - 1) / (2 - 1)
    assert norm_run["q2"]["d2"] == (2 - 1) / (2 - 1)


@pytest.mark.parametrize("norm_name", ["min_max_inverted", "min_max-inverted"])
def test_min_max_norm_with_invert(run, norm_name):
    run_copy = run.run.copy()
    assert run.run == run_copy

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1
    assert run.run["q1"]["d2"] == 2
    assert run.run["q1"]["d3"] == 3
    assert run.run["q2"]["d1"] == 1
    assert run.run["q2"]["d2"] == 2
    assert run.size == 2

    norm_run = norm_switch(norm_name)(run)

    assert len(norm_run) == 2
    assert len(norm_run["q1"]) == 3
    assert len(norm_run["q2"]) == 2
    assert norm_run["q1"]["d1"] == (3 - 1) / (3 - 1)
    assert norm_run["q1"]["d2"] == (3 - 2) / (3 - 1)
    assert norm_run["q1"]["d3"] == (3 - 3) / (3 - 1)
    assert norm_run["q2"]["d1"] == (2 - 1) / (2 - 1)
    assert norm_run["q2"]["d2"] == (2 - 2) / (2 - 1)
