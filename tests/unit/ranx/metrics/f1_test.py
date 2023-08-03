from math import isclose

import pytest

from ranx import Qrels, Run
from ranx.metrics.f1 import f1
from ranx.metrics.precision import precision
from ranx.metrics.recall import recall


# FIXTURES =====================================================================
@pytest.fixture
def qrels_dict():
    return {
        "q1": {
            "d2": 1,  # relevant
            "d5": 1,  # relevant
            "d1": 0,  # non-relevant
            "d4": 0,  # non-relevant
        },
        "q2": {
            "d2": 1,  # relevant
            "d5": 1,  # relevant
            "d1": 0,  # non-relevant
            "d6": 0,  # non-relevant
        },
        "q3": {
            "d2": 1,  # relevant
            "d5": 1,  # relevant
            "d7": 1,  # relevant
            "d1": 0,  # non-relevant
            "d6": 0,  # non-relevant
            "d8": 0,  # non-relevant
        },
        "q4": {
            "d2": 1,  # relevant
            "d4": 1,  # relevant
            "d6": 1,  # relevant
            "d9": 1,  # relevant
            "d1": 0,  # non-relevant
            "d3": 0,  # non-relevant
            "d7": 0,  # non-relevant
            "d8": 0,  # non-relevant
        },
    }


@pytest.fixture
def run_dict():
    return {
        "q1": {
            "d1": 10,  # non-relevant
            "d2": 9,  # relevant
            "d3": 8,  # unjudged
            "d4": 7,  # non-relevant
        },
        "q2": {
            "d1": 10,  # non-relevant
            "d2": 9,  # relevant
            "d3": 8,  # unjudged
            "d4": 7,  # unjudged
            "d5": 6,  # relevant
            "d6": 5,  # non-relevant
        },
        "q3": {
            "d1": 10,  # non-relevant
            "d2": 9,  # relevant
            "d3": 8,  # unjudged
            "d4": 7,  # unjudged
            "d5": 6,  # relevant
            "d6": 5,  # non-relevant
            "d7": 4,  # relevant
            "d8": 3,  # non-relevant
        },
        "q4": {
            "d1": 10,  # non-relevant
            "d12": 9,  # non-relevant
            "d3": 8,  # non-relevant
            "d14": 7,  # non-relevant
            "d16": 5,  # non-relevant
            "d7": 4,  # non-relevant
            "d8": 3,  # non-relevant
        },
    }


@pytest.fixture
def qrels(qrels_dict):
    return Qrels(qrels_dict).to_typed_list()


@pytest.fixture
def run(run_dict):
    return Run(run_dict).to_typed_list()


# TESTS ========================================================================
def test_f1(qrels, run):
    precision_scores = precision(qrels, run)
    recall_scores = recall(qrels, run)
    f1_scores = f1(qrels, run)

    assert len(f1_scores) == len(qrels)

    assert isclose(
        f1_scores[0],
        2
        * (
            (precision_scores[0] * recall_scores[0])
            / (precision_scores[0] + recall_scores[0])
        ),
    )
    assert isclose(
        f1_scores[1],
        2
        * (
            (precision_scores[1] * recall_scores[1])
            / (precision_scores[1] + recall_scores[1])
        ),
    )
    assert isclose(
        f1_scores[2],
        2
        * (
            (precision_scores[2] * recall_scores[2])
            / (precision_scores[2] + recall_scores[2])
        ),
    )
    assert isclose(f1_scores[3], 0.0)
