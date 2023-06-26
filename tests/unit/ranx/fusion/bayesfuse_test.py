from math import isclose

import numpy as np
import pytest
from numba.typed import List as TypedList

from ranx import Qrels, Run
from ranx.fusion import bayesfuse
from ranx.fusion.bayesfuse import _estimate_log_odds, bayesfuse_train
from ranx.metrics import get_hit_lists


# FIXTURES =====================================================================
@pytest.fixture
def qrels():
    qrels = {
        "q1": {"d1": 1, "d2": 1, "d8": 1, "d9": 1, "d10": 1},
        "q2": {"d2": 1, "d4": 1, "d7": 1},
        "q3": {"d1": 1, "d3": 1, "d4": 1, "d5": 1, "d6": 1, "d8": 1, "d9": 1},
    }

    return Qrels(qrels)


@pytest.fixture
def run_1():
    run_dict = {
        "q1": {
            "d1": 10,  # rel
            "d2": 9,  # rel
            "d3": 8,
            "d4": 7,
            "d5": 6,
            "d6": 5,
            "d7": 4,
            "d8": 3,  # rel
            "d9": 2,  # rel
            "d10": 1,  # rel
        },
        "q2": {
            "d1": 10,
            "d2": 9,  # rel
            "d3": 8,
            "d4": 7,  # rel
            "d5": 6,
            "d6": 5,
            "d7": 4,  # rel
            "d8": 3,
            "d9": 2,
            "d10": 1,
        },
        "q3": {
            "d1": 10,  # rel
            "d2": 9,
            "d3": 8,  # rel
            "d4": 7,  # rel
            "d5": 6,  # rel
            "d6": 5,  # rel
            "d7": 4,
            "d8": 3,  # rel
            "d9": 2,  # rel
            "d10": 1,
        },
    }

    return Run(run_dict)


@pytest.fixture
def run_2():
    run_dict = {
        "q1": {
            "d1": 10,  # rel
            "d2": 9,  # rel
            "d8": 8,  # rel
            "d9": 7,  # rel
            "d10": 6,  # rel
            "d3": 5,
            "d4": 4,
            "d5": 3,
            "d6": 2,
            "d7": 1,
        },
        "q2": {
            "d2": 10,  # rel
            "d4": 9,  # rel
            "d7": 8,  # rel
            "d1": 7,
            "d3": 6,
            "d5": 5,
            "d6": 4,
            "d8": 3,
            "d9": 2,
            "d10": 1,
        },
        "q3": {
            "d1": 10,  # rel
            "d2": 9,
            "d3": 8,  # rel
            "d4": 7,  # rel
            "d5": 6,  # rel
            "d6": 5,  # rel
            "d7": 4,
            "d8": 3,  # rel
            "d9": 2,  # rel
            "d10": 1,
        },
    }

    return Run(run_dict)


# TESTS ========================================================================
def test_estimate_log_odds(qrels, run_1, run_2):
    cut_offs = TypedList([0, 5, 10, 15, 20, 30, 100, 200, 500, 1000])

    # Run 1 --------------------------------------------------------------------
    hit_lists = get_hit_lists(qrels.to_typed_list(), run_1.to_typed_list())

    log_odds = _estimate_log_odds(hit_lists, cut_offs)

    p_rel = 8 / 15
    p_nonrel = 1 - p_rel
    assert all(x == np.log(p_rel / p_nonrel) for x in log_odds[:5])

    p_rel = 7 / 15
    p_nonrel = 1 - p_rel
    assert all(x == np.log(p_rel / p_nonrel) for x in log_odds[5:10])

    assert all(x < 0 for x in log_odds[10:])

    # Run 2 --------------------------------------------------------------------
    hit_lists = get_hit_lists(qrels.to_typed_list(), run_2.to_typed_list())

    log_odds = _estimate_log_odds(hit_lists, cut_offs)

    p_rel = 12 / 15
    p_nonrel = 1 - p_rel
    assert all(x == np.log(p_rel / p_nonrel) for x in log_odds[:5])

    p_rel = 3 / 15
    p_nonrel = 1 - p_rel
    assert all(isclose(x, np.log(p_rel / p_nonrel)) for x in log_odds[5:10])

    assert all(x < 0 for x in log_odds[10:])


def test_bayesfuse(qrels, run_1, run_2):
    log_odds = bayesfuse_train(qrels, [run_1, run_2])
    combined_run = bayesfuse([run_1, run_2], log_odds)

    assert combined_run.name == "bayesfuse"

    assert len(combined_run) == 3

    assert "q1" in combined_run.run
    assert "q2" in combined_run.run
    assert "q3" in combined_run.run

    assert len(combined_run["q1"]) == 10
    assert len(combined_run["q2"]) == 10
    assert len(combined_run["q3"]) == 10

    assert isclose(combined_run["q1"]["d1"], log_odds[0][0] + log_odds[1][0])
    assert isclose(combined_run["q1"]["d2"], log_odds[0][1] + log_odds[1][1])
    assert isclose(combined_run["q1"]["d3"], log_odds[0][2] + log_odds[1][5])
    assert isclose(combined_run["q1"]["d4"], log_odds[0][3] + log_odds[1][6])
    assert isclose(combined_run["q1"]["d5"], log_odds[0][4] + log_odds[1][7])
    assert isclose(combined_run["q1"]["d6"], log_odds[0][5] + log_odds[1][8])
    assert isclose(combined_run["q1"]["d7"], log_odds[0][6] + log_odds[1][8])
    assert isclose(combined_run["q1"]["d8"], log_odds[0][7] + log_odds[1][2])
    assert isclose(combined_run["q1"]["d9"], log_odds[0][8] + log_odds[1][3])
    assert isclose(combined_run["q1"]["d10"], log_odds[0][9] + log_odds[1][4])

    assert isclose(combined_run["q2"]["d1"], log_odds[0][0] + log_odds[1][3])
    assert isclose(combined_run["q2"]["d2"], log_odds[0][1] + log_odds[1][0])
    assert isclose(combined_run["q2"]["d3"], log_odds[0][2] + log_odds[1][4])
    assert isclose(combined_run["q2"]["d4"], log_odds[0][3] + log_odds[1][1])
    assert isclose(combined_run["q2"]["d5"], log_odds[0][4] + log_odds[1][5])
    assert isclose(combined_run["q2"]["d6"], log_odds[0][5] + log_odds[1][6])
    assert isclose(combined_run["q2"]["d7"], log_odds[0][6] + log_odds[1][2])
    assert isclose(combined_run["q2"]["d8"], log_odds[0][7] + log_odds[1][7])
    assert isclose(combined_run["q2"]["d9"], log_odds[0][8] + log_odds[1][8])
    assert isclose(combined_run["q2"]["d10"], log_odds[0][9] + log_odds[1][9])

    assert isclose(combined_run["q3"]["d1"], log_odds[0][0] + log_odds[1][0])
    assert isclose(combined_run["q3"]["d2"], log_odds[0][1] + log_odds[1][1])
    assert isclose(combined_run["q3"]["d3"], log_odds[0][2] + log_odds[1][2])
    assert isclose(combined_run["q3"]["d4"], log_odds[0][3] + log_odds[1][3])
    assert isclose(combined_run["q3"]["d5"], log_odds[0][4] + log_odds[1][4])
    assert isclose(combined_run["q3"]["d6"], log_odds[0][5] + log_odds[1][5])
    assert isclose(combined_run["q3"]["d7"], log_odds[0][6] + log_odds[1][6])
    assert isclose(combined_run["q3"]["d8"], log_odds[0][7] + log_odds[1][7])
    assert isclose(combined_run["q3"]["d9"], log_odds[0][8] + log_odds[1][8])
    assert isclose(combined_run["q3"]["d10"], log_odds[0][9] + log_odds[1][9])
