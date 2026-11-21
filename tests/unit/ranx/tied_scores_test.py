from itertools import permutations

import pytest

from ranx import Qrels, Run, evaluate


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize("order", list(permutations(["doc2", "doc10", "doc1"])))
def test_tied_scores_use_descending_string_document_ids(as_object, order):
    qrels = Qrels({"q": {"doc10": 1}})
    scores = {"q": {doc: 0.5 for doc in order}}
    run = Run(scores) if as_object else scores
    # Reverse lexicographic order is doc2, doc10, doc1, not numeric order.
    assert evaluate(qrels, run, "mrr") == 0.5
    assert evaluate(qrels, run, "precision@1") == 0.0


@pytest.mark.parametrize("method", ["add_score", "add"])
def test_incremental_runs_use_the_same_tie_break(method):
    run = Run()
    if method == "add_score":
        run.add_score("q", "doc1", 0.5)
        run.add_score("q", "doc2", 0.5)
    else:
        run.add("q", ["doc1", "doc2"], [0.5, 0.5])
    assert evaluate(Qrels({"q": {"doc1": 1}}), run, "mrr") == 0.5
    assert list(run.to_dict()["q"]) == ["doc2", "doc1"]


@pytest.mark.parametrize("as_object", [False, True])
def test_document_id_tie_break_does_not_override_scores(as_object):
    scores = {"q": {"doc1": 0.8, "doc2": 0.5}}
    run = Run(scores) if as_object else scores
    assert evaluate(Qrels({"q": {"doc1": 1}}), run, "mrr") == 1.0
