# Test rm.py
import numpy as np
import pytest
from numba.typed import List

from ranx import Qrels, Run, evaluate


# BINARY RELEVANCE =============================================================
# hits --------------------------------------------------------------------
def test_hits_single():  # OK
    y_true = np.array([[[1, 1], [4, 1], [5, 1], [6, 1]]])
    y_pred = np.array([[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]]])
    k = 5

    assert evaluate(y_true, y_pred, f"hits@{k}") == 3


def test_hits_parallel():  # OK
    y_true = List(
        [
            np.array([[1, 1], [4, 1], [5, 1], [6, 1]]),
            np.array([[1, 1], [4, 1], [6, 1]]),
        ]
    )
    y_pred = List(
        [
            np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]]),
            np.array([[1, 1], [2, 1], [4, 1], [3, 1], [5, 1], [7, 1]]),
        ]
    )

    k = 5

    assert evaluate(y_true, y_pred, f"hits@{k}") == 2.5


# precision ---------------------------------------------------------------
def test_precision_single():
    y_true = np.array([[[1, 1], [4, 1], [5, 1], [6, 1]]])
    y_pred = np.array([[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]]])
    k = 5
    assert evaluate(y_true, y_pred, f"precision@{k}") == 3 / k


def test_precision_parallel():
    y_true = List(
        [
            np.array([[1, 1], [2, 1], [3, 1]]),
            np.array([[4, 1], [5, 1], [6, 1], [7, 1]]),
            np.array([[8, 1], [9, 1]]),
        ]
    )
    y_pred = List(
        [
            np.array([[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]]),
            np.array([[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]]),
            np.array([[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]]),
        ]
    )

    k = 5

    p_1 = evaluate(List([y_true[0]]), List([y_pred[0]]), f"precision@{k}")
    p_2 = evaluate(List([y_true[1]]), List([y_pred[1]]), f"precision@{k}")
    p_3 = evaluate(List([y_true[2]]), List([y_pred[2]]), f"precision@{k}")

    mean_precision_score = sum([p_1, p_2, p_3]) / 3

    assert np.allclose(evaluate(y_true, y_pred, f"precision@{k}"), mean_precision_score)


# average_precision ------------------------------------------------------------
def test_average_precision():
    y_true = np.array([[[1, 1], [4, 1], [5, 1], [6, 1]]])
    y_pred = np.array([[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]]])

    k = 6

    assert np.allclose(evaluate(y_true, y_pred, f"map@{k}"), 0.525)


# average_precision -------------------------------------------------------
def test_average_precision_parallel():
    y_true = List()
    y_true.append(np.array([[1, 1], [2, 1], [3, 1]]))
    y_true.append(np.array([[4, 1], [5, 1], [6, 1], [7, 1]]))
    y_true.append(np.array([[8, 1], [9, 1]]))
    y_pred = List(
        [
            np.array([[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]]),
            np.array([[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]]),
            np.array([[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]]),
        ]
    )

    k = 7

    ap_1 = evaluate(List([y_true[0]]), List([y_pred[0]]), f"map@{k}")
    ap_2 = evaluate(List([y_true[1]]), List([y_pred[1]]), f"map@{k}")
    ap_3 = evaluate(List([y_true[2]]), List([y_pred[2]]), f"map@{k}")

    assert np.allclose(
        evaluate(y_true, y_pred, f"map@{k}"), sum([ap_1, ap_2, ap_3]) / 3
    )


# reciprocal_rank --------------------------------------------------------
def test_reciprocal_rank_single():
    y_true = np.array([[[3, 1]]])
    y_pred = np.array([[[2, 1], [3, 1], [1, 1], [4, 1], [5, 1]]])

    k = 5

    assert np.allclose(evaluate(y_true, y_pred, f"mrr@{k}"), 0.5)


def test_reciprocal_rank_single_no_match():
    y_true = np.array([[[3, 1]]])
    y_pred = np.array([[[2, 1], [1, 1], [4, 1], [5, 1]]])

    k = 5

    assert np.allclose(evaluate(y_true, y_pred, f"mrr@{k}"), 0.0)


def test_reciprocal_rank_parallel():
    y_true = List()
    y_true.append(np.array([[3, 1]]))
    y_true.append(np.array([[1, 1]]))
    y_true.append(np.array([[3, 1]]))
    y_pred = List(
        [
            np.array([[2, 1], [3, 1], [1, 1], [4, 1], [5, 1]]),
            np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]),
            np.array([[1, 1], [1, 1], [1, 1], [4, 1], [5, 1]]),
        ]
    )

    k = 5

    assert np.allclose(evaluate(y_true, y_pred, f"mrr@{k}"), 0.5)


# r_precision ------------------------------------------------------------------
def test_r_precision_single():
    y_true = np.array([[[1, 1], [2, 1], [3, 1]]])
    y_pred = np.array([[[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]]])

    assert np.allclose(evaluate(y_true, y_pred, "r-precision"), 2 / 3)


def test_r_precision_parallel():
    y_true = List(
        [
            np.array([[1, 1], [2, 1], [3, 1]]),
            np.array([[1, 1], [2, 1]]),
        ]
    )
    y_pred = List(
        [
            np.array([[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]]),
            np.array([[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]]),
        ]
    )

    assert np.allclose(evaluate(y_true, y_pred, "r-precision"), (2 / 3 + 1 / 2) / 2)


# recall ------------------------------------------------------------------
def test_recall_single():
    y_true = np.array([[[1, 1], [2, 1], [3, 1]]])
    y_pred = np.array([[[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]]])
    k = 2

    assert np.allclose(evaluate(y_true, y_pred, f"recall@{k}"), 1 / 3)


def test_recall_parallel():
    y_true = List()
    y_true.append(np.array([[1, 1], [2, 1], [3, 1]]))
    y_true.append(np.array([[1, 1], [2, 1]]))
    y_pred = List(
        [
            np.array([[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]]),
            np.array([[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]]),
        ]
    )
    k = 2

    assert np.allclose(evaluate(y_true, y_pred, f"recall@{k}"), (1 / 3 + 1 / 2) / 2)


# # NON-BINARY RELEVANCE rm =================================================
def test_ndcg_jarvelin():
    # List of IDs ordered by descending order of true relevance
    y_true = np.array([[[2, 5], [4, 4], [5, 3], [10, 2]]])
    # List of IDs orderd by descending order of predicted relevance
    y_pred_1 = np.array(
        [[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]]
    )  # rel = 0, 5, 0, 4, 3
    y_pred_2 = np.array(
        [[[10, 1], [5, 1], [2, 1], [4, 1], [3, 1]]]
    )  # rel = 2, 3, 5, 4, 0
    y_pred_3 = np.array(
        [[[1, 1], [3, 1], [6, 1], [7, 1], [8, 1]]]
    )  # rel = 0, 0, 0, 0, 0

    idcg = 5 / np.log2(2) + 4 / np.log2(3) + 3 / np.log2(4) + 2 / np.log2(5)

    k = 10

    assert np.allclose(
        evaluate(y_true, y_pred_1, f"ndcg@{k}"),
        (5 / np.log2(3) + 4 / np.log2(5) + 3 / np.log2(6)) / idcg,
    )

    assert np.allclose(
        evaluate(y_true, y_pred_2, f"ndcg@{k}"),
        (2 / np.log2(2) + 3 / np.log2(3) + 5 / np.log2(4) + 4 / np.log2(5)) / idcg,
    )

    assert np.allclose(evaluate(y_true, y_pred_3, f"ndcg@{k}"), 0.0)


def test_ndcg_burges():
    # List of IDs ordered by descending order of true relevance
    y_true = np.array([[[2, 5], [4, 4], [5, 3], [10, 2]]])
    # List of IDs orderd by descending order of predicted relevance
    y_pred_1 = np.array(
        [[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]]
    )  # rel = 0, 5, 0, 4, 3
    y_pred_2 = np.array(
        [[[10, 1], [5, 1], [2, 1], [4, 1], [3, 1]]]
    )  # rel = 2, 3, 5, 4, 0
    y_pred_3 = np.array(
        [[[1, 1], [3, 1], [6, 1], [7, 1], [8, 1]]]
    )  # rel = 0, 0, 0, 0, 0

    idcg = (
        (2 ** 5 - 1) / np.log2(2)
        + (2 ** 4 - 1) / np.log2(3)
        + (2 ** 3 - 1) / np.log2(4)
        + (2 ** 2 - 1) / np.log2(5)
    )

    k = 10

    assert np.allclose(
        evaluate(y_true, y_pred_1, f"ndcg_burges@{k}"),
        (
            (2 ** 5 - 1) / np.log2(3)
            + (2 ** 4 - 1) / np.log2(5)
            + (2 ** 3 - 1) / np.log2(6)
        )
        / idcg,
    )

    assert np.allclose(
        evaluate(y_true, y_pred_2, f"ndcg_burges@{k}"),
        (
            (2 ** 2 - 1) / np.log2(2)
            + (2 ** 3 - 1) / np.log2(3)
            + (2 ** 5 - 1) / np.log2(4)
            + (2 ** 4 - 1) / np.log2(5)
        )
        / idcg,
    )

    assert np.allclose(evaluate(y_true, y_pred_3, f"ndcg_burges@{k}"), 0.0)


def test_with_Qrels_and_Run_control():
    # Create empty Qrels
    qrels = Qrels()
    # Add queries to qrels
    qrels.add_multi(
        q_ids=["q_1", "q_2"],
        doc_ids=[
            ["doc_12", "doc_25"],  # q_1 relevant documents
            ["doc_11", "doc_2"],  # q_2 relevant documents
        ],
        scores=[
            [5, 3],  # q_1 relevance judgements
            [6, 1],  # q_2 relevance judgements
        ],
    )

    # Create empty Run
    run = Run()
    # Add queries to run
    run.add_multi(
        q_ids=["q_1", "q_2"],
        doc_ids=[
            # q_1 retrieved documents
            ["doc_12", "doc_23", "doc_25", "doc_36", "doc_32", "doc_35"],
            # q_2 retrieved documents
            ["doc_12", "doc_11", "doc_25", "doc_36", "doc_2", "doc_35"],
        ],
        scores=[
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # q_1 retrieved document scores
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # q_2 retrieved document scores
        ],
    )

    evaluate(qrels, run, "ndcg@5")


def test_python_dict():
    qrels = {
        "q_1": {
            "doc_12": 5,
            "doc_25": 3,
        },
        "q_2": {
            "doc_11": 6,
            "doc_2": 1,
        },
    }
    run = {
        "q_1": {
            "doc_12": 0.9,
            "doc_23": 0.8,
            "doc_25": 0.7,
            "doc_36": 0.6,
            "doc_32": 0.5,
            "doc_35": 0.4,
        },
        "q_2": {
            "doc_12": 0.9,
            "doc_11": 0.8,
            "doc_25": 0.7,
            "doc_36": 0.6,
            "doc_2": 0.5,
            "doc_35": 0.4,
        },
    }

    evaluate(qrels, run, "ndcg@5")


def test_python_dict_2():

    # Create empty Qrels
    qrels = Qrels()
    # Add queries to qrels
    qrels.add_multi(
        q_ids=["q_1", "q_2"],
        doc_ids=[
            ["doc_12", "doc_25"],  # q_1 relevant documents
            ["doc_11", "doc_2"],  # q_2 relevant documents
        ],
        scores=[
            [5, 3],  # q_1 relevance judgements
            [6, 1],  # q_2 relevance judgements
        ],
    )

    # Create empty Run
    run = Run()
    # Add queries to run
    run.add_multi(
        q_ids=["q_1", "q_2"],
        doc_ids=[
            # q_1 retrieved documents
            ["doc_12", "doc_23", "doc_25", "doc_36", "doc_32", "doc_35"],
            # q_2 retrieved documents
            ["doc_12", "doc_11", "doc_25", "doc_36", "doc_2", "doc_35"],
        ],
        scores=[
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # q_1 retrieved document scores
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # q_2 retrieved document scores
        ],
    )

    x = evaluate(qrels, run, "ndcg@5")

    qrels = {
        "q_1": {
            "doc_12": 5,
            "doc_25": 3,
        },
        "q_2": {
            "doc_11": 6,
            "doc_2": 1,
        },
    }
    run = {
        "q_1": {
            "doc_35": 0.4,
            "doc_25": 0.7,
            "doc_23": 0.8,
            "doc_36": 0.6,
            "doc_12": 0.9,
            "doc_32": 0.5,
        },
        "q_2": {
            "doc_12": 0.9,
            "doc_25": 0.7,
            "doc_2": 0.5,
            "doc_36": 0.6,
            "doc_11": 0.8,
            "doc_35": 0.4,
        },
    }

    y = evaluate(qrels, run, "ndcg@5")

    assert x == y


def test_keys_control():
    # Create empty Qrels
    qrels = Qrels()
    # Add queries to qrels
    qrels.add_multi(
        q_ids=["q_1", "q_2"],
        doc_ids=[
            ["doc_12", "doc_25"],  # q_1 relevant documents
            ["doc_11", "doc_2"],  # q_2 relevant documents
        ],
        scores=[
            [5, 3],  # q_1 relevance judgements
            [6, 1],  # q_2 relevance judgements
        ],
    )

    # Create empty Run
    run = Run()
    # Add queries to run
    run.add_multi(
        q_ids=["q_1", "q_3"],
        doc_ids=[
            # q_1 retrieved documents
            ["doc_12", "doc_23", "doc_25", "doc_36", "doc_32", "doc_35"],
            # q_2 retrieved documents
            ["doc_12", "doc_11", "doc_25", "doc_36", "doc_2", "doc_35"],
        ],
        scores=[
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # q_1 retrieved document scores
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # q_2 retrieved document scores
        ],
    )

    with pytest.raises(Exception):
        evaluate(qrels, run, "ndcg@5")
