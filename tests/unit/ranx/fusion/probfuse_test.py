import numpy as np
import pytest
from ranx import Qrels, Run
from ranx.fusion import probfuse, probfuse_train


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
def test_probfuse_train(qrels, run_1, run_2, run_3):
    probs = probfuse_train(qrels, [run_1, run_2, run_3], 2)

    # Probs run_1
    assert probs[0].tolist() == [0.0, 1.0]

    # Probs run_2
    assert probs[1].tolist() == [0.5, 0.5]

    # Probs run_3
    assert probs[2].tolist() == [0, 0]


def test_probfuse(qrels, run_1, run_2, run_3):
    probs = probfuse_train(qrels, [run_1, run_2, run_3], 2)
    combined_run = probfuse([run_1, run_2, run_3], probs)

    assert combined_run.name == "probfuse"

    assert len(combined_run) == 2
    assert len(combined_run["q1"]) == 3
    assert len(combined_run["q2"]) == 3

    assert combined_run["q1"]["d1"] == probs[0][1] / 2 + probs[1][0]
    assert combined_run["q1"]["d2"] == probs[0][0] + probs[1][1] / 2
    assert combined_run["q1"]["d3"] == probs[0][0] + probs[2][0]

    assert combined_run["q2"]["d1"] == probs[0][1] / 2 + probs[1][1] / 2
    assert combined_run["q2"]["d2"] == probs[0][0] + probs[2][1] / 2
    assert combined_run["q2"]["d3"] == probs[1][0] + probs[2][0]
