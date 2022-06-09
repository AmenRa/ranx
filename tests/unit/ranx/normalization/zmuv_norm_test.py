import pytest
from ranx import Run
from ranx.normalization import zmuv_norm


# FIXTURES =====================================================================
@pytest.fixture
def run():
    run_dict = {
        "q1": {"d1": 1, "d2": 2, "d3": 3},
        "q2": {"d1": 1, "d2": 2},
    }

    return Run(run_dict)


# TESTS ========================================================================
def test_zmuv_norm(run):
    run_copy = run.run.copy()

    norm_run = zmuv_norm(run)

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
    assert norm_run["q1"]["d1"] == (1 - 2) / 0.816496580927726
    assert norm_run["q1"]["d2"] == (2 - 2) / 0.816496580927726
    assert norm_run["q1"]["d3"] == (3 - 2) / 0.816496580927726
    assert norm_run["q2"]["d1"] == (1 - 1.5) / 0.5
    assert norm_run["q2"]["d2"] == (2 - 1.5) / 0.5
