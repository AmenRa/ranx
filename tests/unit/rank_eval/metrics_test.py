# Test rm.py
import numpy as np
import pytest
from numba.typed import List

from rank_eval import metrics as rm


def test_choose_optimal_function_parallel():
    y_true = List()
    y_pred = []
    y_t_1 = np.array([[1, 1], [4, 1], [5, 1], [6, 1]])
    y_p_1 = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [1, 7]])
    y_t_2 = np.array([[1, 1], [4, 1], [5, 1], [6, 1]])
    y_p_2 = np.array([[1, 1], [2, 1], [4, 1], [3, 1], [5, 1], [1, 7]])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)

    k = 5

    with pytest.raises(TypeError) as e:
        rm._choose_optimal_function(
            y_true=y_true,
            y_pred=y_pred,
            f_name="hits_at_k",
            f_single=rm._hits_at_k,
            f_parallel=rm._hits_at_k_parallel,
            f_additional_args={"k": k},
        )
        assert "y_pred type not supported." in str(e.value)

    res = rm._choose_optimal_function(
        y_true=y_true,
        y_pred=np.array(y_pred),
        f_name="hits_at_k",
        f_single=rm._hits_at_k,
        f_parallel=rm._hits_at_k_parallel,
        f_additional_args={"k": k},
    )

    assert res == 3

    res = rm._choose_optimal_function(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        f_name="hits_at_k",
        f_single=rm._hits_at_k,
        f_parallel=rm._hits_at_k_parallel,
        f_additional_args={"k": k},
    )

    assert res == 3


def test_choose_optimal_function_parallel():
    y_true = []
    y_pred = []
    y_t_1 = np.array([[1, 1], [4, 1], [5, 1], [6, 1]])
    y_p_1 = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [1, 7]])
    y_t_2 = np.array([[1, 1], [4, 1], [5, 1], [6, 1]])
    y_p_2 = np.array([[1, 1], [2, 1], [4, 1], [3, 1], [5, 1], [1, 7]])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)

    k = 5

    res = rm._choose_optimal_function(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        f_name="hits_at_k",
        f_single=rm._hits_at_k,
        f_parallel=rm._hits_at_k_parallel,
        f_additional_args={"k": k},
    )

    assert res == 3


def test_choose_optimal_function_single():
    y_true = [[1, 1], [4, 1], [5, 1], [6, 1]]
    y_pred = [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [1, 7]]

    k = 5

    res = rm._choose_optimal_function(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        f_name="hits_at_k",
        f_single=rm._hits_at_k,
        f_parallel=rm._hits_at_k_parallel,
        f_additional_args={"k": k},
    )

    assert res == 3


# BINARY RELEVANCE =============================================================
# hits_at_k --------------------------------------------------------------------
def test_hits_at_k_single():  # OK
    y_true = np.array([[1, 1], [4, 1], [5, 1], [6, 1]])
    y_pred = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]])
    k = 5
    assert rm.hits_at_k(y_true, y_pred, k) == 3


def test_hits_at_k_parallel():  # OK
    y_true = List()
    y_pred = []
    y_t_1 = np.array([[1, 1], [4, 1], [5, 1], [6, 1]])
    y_p_1 = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]])
    y_t_2 = np.array([[1, 1], [4, 1], [6, 1]])
    y_p_2 = np.array([[1, 1], [2, 1], [4, 1], [3, 1], [5, 1], [7, 1]])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)
    y_pred = np.array(y_pred)

    k = 5

    assert rm.hits_at_k(y_true, y_pred, k) == 2.5


# precision_at_k ---------------------------------------------------------------
def test_precision_at_k_single():
    y_true = np.array([[1, 1], [4, 1], [5, 1], [6, 1]])
    y_pred = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]])
    k = 5
    assert rm.precision_at_k(y_true, y_pred, k) == 3 / k


def test_precision_at_k_parallel():
    y_true = List()
    y_true.append(np.array([[1, 1], [2, 1], [3, 1]]))
    y_true.append(np.array([[4, 1], [5, 1], [6, 1], [7, 1]]))
    y_true.append(np.array([[8, 1], [9, 1]]))
    y_pred = np.array(
        [
            [[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]],
            [[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]],
            [[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]],
        ]
    )

    k = 5

    p_1 = rm.precision_at_k(y_true[0], y_pred[0], k)
    p_2 = rm.precision_at_k(y_true[1], y_pred[1], k)
    p_3 = rm.precision_at_k(y_true[2], y_pred[2], k)

    mean_precision_at_k_score = sum([p_1, p_2, p_3]) / 3

    assert np.allclose(rm.precision_at_k(y_true, y_pred, k), mean_precision_at_k_score)


# average_precision ------------------------------------------------------------
def test_average_precision():
    y_true = np.array([[1, 1], [4, 1], [5, 1], [6, 1]])
    y_pred = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]])

    k = 6

    assert np.allclose(rm.map(y_true, y_pred, k), 0.525)


# mean_average_precision -------------------------------------------------------
def test_map_parallel():
    y_true = List()
    y_true.append(np.array([[1, 1], [2, 1], [3, 1]]))
    y_true.append(np.array([[4, 1], [5, 1], [6, 1], [7, 1]]))
    y_true.append(np.array([[8, 1], [9, 1]]))
    y_pred = np.array(
        [
            [[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]],
            [[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]],
            [[4, 1], [5, 1], [6, 1], [2, 1], [1, 1], [7, 1], [3, 1]],
        ]
    )

    k = 7

    ap_1 = rm.map(y_true[0], y_pred[0], k)
    ap_2 = rm.map(y_true[1], y_pred[1], k)
    ap_3 = rm.map(y_true[2], y_pred[2], k)

    assert np.allclose(rm.map(y_true, y_pred, k), sum([ap_1, ap_2, ap_3]) / 3)


# mean_reciprocal_rank --------------------------------------------------------
def test_mrr_single():
    y_true = np.array([[3, 1]])
    y_pred = np.array([[2, 1], [3, 1], [1, 1], [4, 1], [5, 1]])

    k = 5

    assert np.allclose(rm.mrr(y_true, y_pred, k), 0.5)


def test_mrr_single_no_match():
    y_true = np.array([[3, 1]])
    y_pred = np.array([[2, 1], [1, 1], [4, 1], [5, 1]])

    k = 5

    assert np.allclose(rm.mrr(y_true, y_pred, k), 0.0)


def test_mrr_parallel():
    y_true = List()
    y_true.append(np.array([[3, 1]]))
    y_true.append(np.array([[1, 1]]))
    y_true.append(np.array([[3, 1]]))
    y_pred = np.array(
        [
            [[2, 1], [3, 1], [1, 1], [4, 1], [5, 1]],
            [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]],
            [[1, 1], [1, 1], [1, 1], [4, 1], [5, 1]],
        ]
    )

    k = 5

    assert np.allclose(rm.mrr(y_true, y_pred, k), 0.5)


# r_precision ------------------------------------------------------------------
def test_r_precision_single():
    y_true = np.array([[1, 1], [2, 1], [3, 1]])
    y_pred = np.array([[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]])

    assert np.allclose(rm.r_precision(y_true, y_pred), 2 / 3)


def test_r_precision_parallel():
    y_true = List()
    y_true.append(np.array([[1, 1], [2, 1], [3, 1]]))
    y_true.append(np.array([[1, 1], [2, 1]]))
    y_pred = np.array(
        [
            [[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]],
            [[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]],
        ]
    )

    assert np.allclose(rm.r_precision(y_true, y_pred), (2 / 3 + 1 / 2) / 2)


# recall_at_k ------------------------------------------------------------------
def test_recall_at_k_single():
    y_true = np.array([[1, 1], [2, 1], [3, 1]])
    y_pred = np.array([[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]])
    k = 2

    assert np.allclose(rm.recall_at_k(y_true, y_pred, k), 1 / 3)


def test_recall_at_k_parallel():
    y_true = List()
    y_true.append(np.array([[1, 1], [2, 1], [3, 1]]))
    y_true.append(np.array([[1, 1], [2, 1]]))
    y_pred = np.array(
        [
            [[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]],
            [[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]],
        ]
    )
    k = 2

    assert np.allclose(rm.recall_at_k(y_true, y_pred, k), (1 / 3 + 1 / 2) / 2)


# # NON-BINARY RELEVANCE rm =================================================
def test_ndcg():
    # List of IDs ordered by descending order of true relevance
    y_true = np.array([[2, 5], [4, 4], [5, 3], [10, 2]])
    # List of IDs orderd by descending order of predicted relevance
    y_pred_1 = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]])  # rel = 0, 5, 0, 4, 3
    y_pred_2 = np.array(
        [[10, 1], [5, 1], [2, 1], [4, 1], [3, 1]]
    )  # rel = 2, 3, 5, 4, 0
    y_pred_3 = np.array([[1, 1], [3, 1], [6, 1], [7, 1], [8, 1]])  # rel = 0, 0, 0, 0, 0

    idcg = (
        (2 ** 5 - 1) / np.log2(2)
        + (2 ** 4 - 1) / np.log2(3)
        + (2 ** 3 - 1) / np.log2(4)
        + (2 ** 2 - 1) / np.log2(5)
    )

    k = 10

    assert np.allclose(
        rm.ndcg(y_true, y_pred_1, k),
        (
            (2 ** 5 - 1) / np.log2(3)
            + (2 ** 4 - 1) / np.log2(5)
            + (2 ** 3 - 1) / np.log2(6)
        )
        / idcg,
    )

    assert np.allclose(
        rm.ndcg(y_true, y_pred_2, k),
        (
            (2 ** 2 - 1) / np.log2(2)
            + (2 ** 3 - 1) / np.log2(3)
            + (2 ** 5 - 1) / np.log2(4)
            + (2 ** 4 - 1) / np.log2(5)
        )
        / idcg,
    )

    assert np.allclose(rm.ndcg(y_true, y_pred_3, k), 0.0)
