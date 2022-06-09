import numpy as np
import pytest
from numba.typed import List

import ranx.metrics as rm


# BINARY RELEVANCE =============================================================
# hits --------------------------------------------------------------------
def test_hits():  # OK
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

    assert np.mean(rm.hits(y_true, y_pred, k)) == 2.5


# hit_rate --------------------------------------------------------------------
def test_hit_rate():  # OK
    y_true = List()
    y_pred = []
    y_t_1 = np.array([[9, 1], [6, 1], [8, 1], [6, 1]])
    y_p_1 = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [7, 1]])
    y_t_2 = np.array([[1, 1], [2, 1], [4, 1]])
    y_p_2 = np.array([[1, 1], [2, 1], [4, 1], [3, 1], [5, 1], [7, 1]])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)
    y_pred = np.array(y_pred)

    k = 5

    assert np.mean(rm.hit_rate(y_true, y_pred, k)) == 0.5


# precision ---------------------------------------------------------------
def test_precision():
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

    p_1 = rm.precision(List([y_true[0]]), List([y_pred[0]]), k)[0]
    p_2 = rm.precision(List([y_true[1]]), List([y_pred[1]]), k)[0]
    p_3 = rm.precision(List([y_true[2]]), List([y_pred[2]]), k)[0]

    assert np.allclose(
        np.mean(rm.precision(y_true, y_pred, k)), sum([p_1, p_2, p_3]) / 3
    )


# average_precision -------------------------------------------------------
def test_average_precision():
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

    ap_1 = rm.average_precision(List([y_true[0]]), List([y_pred[0]]), k)[0]
    ap_2 = rm.average_precision(List([y_true[1]]), List([y_pred[1]]), k)[0]
    ap_3 = rm.average_precision(List([y_true[2]]), List([y_pred[2]]), k)[0]

    assert np.allclose(
        np.mean(rm.average_precision(y_true, y_pred, k)),
        sum([ap_1, ap_2, ap_3]) / 3,
    )


# reciprocal_rank --------------------------------------------------------
def test_reciprocal_rank_single():
    y_true = np.array([[[3, 1]]])
    y_pred = np.array([[[2, 1], [3, 1], [1, 1], [4, 1], [5, 1]]])

    k = 5

    assert np.allclose(rm.reciprocal_rank(y_true, y_pred, k)[0], 0.5)


def test_reciprocal_rank_single_no_match():
    y_true = np.array([[[3, 1]]])
    y_pred = np.array([[[2, 1], [1, 1], [4, 1], [5, 1]]])

    k = 5

    assert np.allclose(rm.reciprocal_rank(y_true, y_pred, k)[0], 0.0)


def test_reciprocal_rank():
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

    assert np.allclose(np.mean(rm.reciprocal_rank(y_true, y_pred, k)), 0.5)


# r_precision ------------------------------------------------------------------
def test_r_precision_single():
    y_true = np.array([[[1, 1], [2, 1], [3, 1]]])
    y_pred = np.array(
        [[[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]]]
    )

    assert np.allclose(rm.r_precision(y_true, y_pred)[0], 2 / 3)


def test_r_precision():
    y_true = List()
    y_true.append(np.array([[1, 1], [2, 1], [3, 1]]))
    y_true.append(np.array([[1, 1], [2, 1]]))
    y_pred = np.array(
        [
            [[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]],
            [[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]],
        ]
    )

    assert np.allclose(
        np.mean(rm.r_precision(y_true, y_pred)), (2 / 3 + 1 / 2) / 2
    )


# recall ------------------------------------------------------------------
def test_recall_single():
    y_true = np.array([[[1, 1], [2, 1], [3, 1]]])
    y_pred = np.array(
        [[[2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [6, 1], [7, 1]]]
    )
    k = 2

    assert np.allclose(rm.recall(y_true, y_pred, k)[0], 1 / 3)


def test_recall():
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

    assert np.allclose(
        np.mean(rm.recall(y_true, y_pred, k)), (1 / 3 + 1 / 2) / 2
    )


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
        rm.ndcg(y_true, y_pred_1, k)[0],
        (5 / np.log2(3) + 4 / np.log2(5) + 3 / np.log2(6)) / idcg,
    )

    assert np.allclose(
        rm.ndcg(y_true, y_pred_2, k)[0],
        (2 / np.log2(2) + 3 / np.log2(3) + 5 / np.log2(4) + 4 / np.log2(5))
        / idcg,
    )

    assert np.allclose(rm.ndcg(y_true, y_pred_3, k)[0], 0.0)


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
        rm.ndcg_burges(y_true, y_pred_1, k)[0],
        (
            (2 ** 5 - 1) / np.log2(3)
            + (2 ** 4 - 1) / np.log2(5)
            + (2 ** 3 - 1) / np.log2(6)
        )
        / idcg,
    )

    assert np.allclose(
        rm.ndcg_burges(y_true, y_pred_2, k)[0],
        (
            (2 ** 2 - 1) / np.log2(2)
            + (2 ** 3 - 1) / np.log2(3)
            + (2 ** 5 - 1) / np.log2(4)
            + (2 ** 4 - 1) / np.log2(5)
        )
        / idcg,
    )

    assert np.allclose(rm.ndcg(y_true, y_pred_3, k)[0], 0.0)
