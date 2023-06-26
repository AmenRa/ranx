from math import isclose

import pytest
import pytrec_eval

from ranx import Qrels, Run
from ranx.metrics.bpref import bpref


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
            "d2": 9,  # relevant
            "d3": 8,  # non-relevant
            "d4": 7,  # relevant
            "d6": 5,  # relevant
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
def test_bpref(qrels, run):
    scores = bpref(qrels, run)

    assert len(scores) == len(qrels)

    assert isclose(scores[0], 1 / 2 * (1 - (1 / 2)))
    assert isclose(scores[1], 1 / 2 * ((1 - (1 / 2)) + (1 - (1 / 2))))
    assert isclose(scores[2], 1 / 3 * ((1 - (1 / 3)) + (1 - (1 / 3)) + (1 - (2 / 3))))
    assert isclose(scores[3], 1 / 4 * ((1 - (1 / 4)) + (1 - (2 / 4)) + (1 - (2 / 4))))


def test_bpref_vs_trec_eval(qrels_dict, run_dict, qrels, run):
    scores = bpref(qrels, run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, ["bpref"])
    trec_results = evaluator.evaluate(run_dict)

    assert isclose(scores[0], trec_results["q1"]["bpref"])
    assert isclose(scores[1], trec_results["q2"]["bpref"])
    assert isclose(scores[2], trec_results["q3"]["bpref"])
    assert isclose(scores[3], trec_results["q4"]["bpref"])
