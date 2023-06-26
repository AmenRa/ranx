import numpy as np
import pytest

from ranx import Qrels, Run
from ranx.fusion import mapfuse, mapfuse_train


# FIXTURES =====================================================================
@pytest.fixture
def qrels():
    qrels_dict = {
        "q1": {"d1": 1},
        "q2": {"d1": 1},
    }

    return Qrels(qrels_dict)


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
def test_mapfuse(qrels, run_1, run_2, run_3):
    map_scores = mapfuse_train(qrels, [run_1, run_2, run_3])
    combined_run = mapfuse([run_1, run_2, run_3], map_scores)

    assert combined_run.name == "mapfuse"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == map_scores[0] / 3 + map_scores[1] / 1
    assert combined_run["q1"]["d2"] == map_scores[0] / 2 + map_scores[1] / 2
    assert combined_run["q1"]["d3"] == map_scores[0] / 1 + map_scores[2] / 1

    assert combined_run["q2"]["d1"] == map_scores[0] / 2 + map_scores[1] / 2
    assert combined_run["q2"]["d2"] == map_scores[0] / 1 + map_scores[2] / 2
    assert combined_run["q2"]["d3"] == map_scores[1] / 1 + map_scores[2] / 1
