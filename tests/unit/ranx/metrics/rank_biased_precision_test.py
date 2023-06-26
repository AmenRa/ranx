import pytest

from ranx import Qrels, Run
from ranx.metrics.rank_biased_precision import rank_biased_precision


# FIXTURES =====================================================================
@pytest.fixture
def qrels_dict():
    return {
        "q1": {
            "d1": 1,
            "d2": 1,
            "d6": 1,
            "d11": 1,
            "d17": 1,
        },
    }


@pytest.fixture
def run_dict():
    return {
        "q1": {
            "d1": 21 - 1,
            "d2": 21 - 2,
            "d3": 21 - 3,
            "d4": 21 - 4,
            "d5": 21 - 5,
            "d6": 21 - 6,
            "d7": 21 - 7,
            "d8": 21 - 8,
            "d9": 21 - 9,
            "d10": 21 - 10,
            "d11": 21 - 11,
            "d12": 21 - 12,
            "d13": 21 - 13,
            "d14": 21 - 14,
            "d15": 21 - 15,
            "d16": 21 - 16,
            "d17": 21 - 17,
            "d18": 21 - 18,
            "d19": 21 - 19,
            "d20": 21 - 20,
        },
    }


@pytest.fixture
def qrels(qrels_dict):
    return Qrels(qrels_dict).to_typed_list()


@pytest.fixture
def run(run_dict):
    return Run(run_dict).to_typed_list()


# TESTS ========================================================================
def test_rank_biased_precision(qrels, run):
    scores = rank_biased_precision(qrels, run, 0.5)
    assert len(scores) == len(qrels)
    assert round(scores[0], 4) == 0.7661

    scores = rank_biased_precision(qrels, run, 0.8)
    assert len(scores) == len(qrels)
    assert round(scores[0], 4) == 0.4526

    scores = rank_biased_precision(qrels, run, 0.95)
    assert len(scores) == len(qrels)
    assert round(scores[0], 4) == 0.1881
