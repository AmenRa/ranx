import numpy as np
import pytest

from ranx import Qrels, Run
from ranx.metrics.f1 import f1
from ranx.metrics.precision import precision
from ranx.metrics.recall import recall


# FIXTURES =====================================================================
@pytest.fixture
def qrels_dict():
    return {
        "q_1": {"doc_a": 1},
        "q_2": {"doc_b": 1},
        "q_3": {"doc_c": 1},
        "q_4": {"doc_d": 1},
    }


@pytest.fixture
def run_dict():
    return {
        "q_1": {"doc_a": 1.0},
        "q_2": {"doc_b": 1.0},
        "q_3": {},
        "q_4": {"doc_d": 1.0},
    }


@pytest.fixture
def qrels(qrels_dict):
    return Qrels(qrels_dict).to_typed_list()


@pytest.fixture
def run(run_dict):
    return Run(run_dict).to_typed_list()


# TESTS ========================================================================
def test_precision(qrels, run):
    scores = precision(qrels, run)
    assert len(scores) == len(qrels)
    assert np.mean(scores) == 0.75


def test_recall(qrels, run):
    scores = recall(qrels, run)
    assert len(scores) == len(qrels)
    assert np.mean(scores) == 0.75


def test_f1(qrels, run):
    scores = f1(qrels, run)
    assert len(scores) == len(qrels)
    assert np.mean(scores) == 0.75
