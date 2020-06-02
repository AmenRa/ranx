# Test metrics.py
import pytest
import numpy as np
from metrics import ranking_metrics as metrics
from numba.typed import List


def test_choose_optimal_function_parallel():
    y_true = List()
    y_pred = []
    y_t_1 = np.array([1, 4, 5, 6])
    y_p_1 = np.array([1, 2, 3, 4, 5, 7])
    y_t_2 = np.array([1, 4, 5, 6])
    y_p_2 = np.array([1, 2, 4, 3, 5, 7])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)

    k = 5

    with pytest.raises(TypeError) as e:
        metrics._choose_optimal_function(
            y_true,
            y_pred,
            metrics._hits_at_k,
            metrics._hits_at_k_parallel,
            {"k": k},
        )
        assert "y_pred type not supported." in str(e.value)

    res = metrics._choose_optimal_function(
        y_true,
        np.array(y_pred),
        metrics._hits_at_k,
        metrics._hits_at_k_parallel,
        {"k": k},
    )

    assert res == 3

    res = metrics._choose_optimal_function(
        np.array(y_true),
        np.array(y_pred),
        metrics._hits_at_k,
        metrics._hits_at_k_parallel,
        {"k": k},
    )

    assert res == 3


def test_choose_optimal_function_multi():
    y_true = []
    y_pred = []
    y_t_1 = np.array([1, 4, 5, 6])
    y_p_1 = np.array([1, 2, 3, 4, 5, 7])
    y_t_2 = np.array([1, 4, 5, 6])
    y_p_2 = np.array([1, 2, 4, 3, 5, 7])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)

    k = 5

    res = metrics._choose_optimal_function(
        y_true,
        y_pred,
        metrics._hits_at_k,
        metrics._hits_at_k_parallel,
        {"k": k},
    )

    assert res == 3

    res = metrics._choose_optimal_function(
        y_true,
        np.array(y_pred),
        metrics._hits_at_k,
        metrics._hits_at_k_parallel,
        {"k": k},
    )

    assert res == 3


def test_choose_optimal_function_single():
    y_true = [1, 4, 5, 6]
    y_pred = [1, 2, 3, 4, 5, 7]

    k = 5

    with pytest.raises(TypeError) as e:
        metrics._choose_optimal_function(
            y_true,
            y_pred,
            metrics._hits_at_k,
            metrics._hits_at_k_parallel,
            {"k": k},
        )
        assert "y_true type not supported." in str(e.value)

    with pytest.raises(TypeError) as e:
        metrics._choose_optimal_function(
            np.array(y_true),
            y_pred,
            metrics._hits_at_k,
            metrics._hits_at_k_parallel,
            {"k": k},
        )
        assert "y_true type not supported." in str(e.value)

    res = metrics._choose_optimal_function(
        np.array(y_true),
        np.array(y_pred),
        metrics._hits_at_k,
        metrics._hits_at_k_parallel,
        {"k": k},
    )

    assert res == 3


# BINARY RELEVANCE METRICS =====================================================
# hit_list_at_k ----------------------------------------------------------------
def test_hit_list_at_k():
    y_true = np.array([1, 4, 5, 6])
    y_pred = np.array([1, 2, 3, 4, 5, 7])
    k = 5
    assert np.array_equal(
        metrics.hit_list_at_k(y_true, y_pred, k), [1, 0, 0, 1, 1]
    )


def test_hit_list_at_k_with_k_greater_than_y_pred():
    y_true = np.array([1, 4, 5, 6])
    y_pred = np.array([1, 2, 3, 4, 5, 7])
    k = 10
    assert np.array_equal(
        metrics.hit_list_at_k(y_true, y_pred, k), [1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    )


def test_hit_list_at_k_multi():
    y_true = []
    y_pred = []
    y_t_1 = np.array([1, 4, 5, 6])
    y_p_1 = np.array([1, 2, 3, 4, 5, 7])
    y_t_2 = np.array([1, 4, 5, 6])
    y_p_2 = np.array([1, 2, 4, 3, 5, 7])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)

    k = 5

    res = metrics.hit_list_at_k(y_true, y_pred, k)

    assert np.array_equal(res[0], [1, 0, 0, 1, 1])
    assert np.array_equal(res[1], [1, 0, 1, 0, 1])


# hits_at_k --------------------------------------------------------------------
def test_hits_at_k_single():
    y_true = np.array([1, 4, 5, 6])
    y_pred = np.array([1, 2, 3, 4, 5, 7])
    k = 5
    assert metrics.hits_at_k(y_true, y_pred, k) == 3


def test_hits_at_k_multi():
    y_true = []
    y_pred = []
    y_t_1 = np.array([1, 4, 5, 6])
    y_p_1 = np.array([1, 2, 3, 4, 5, 7])
    y_t_2 = np.array([1, 4, 5, 6])
    y_p_2 = np.array([1, 2, 4, 3, 5, 7])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)

    k = 5

    assert metrics.hits_at_k(y_true, y_pred, k) == 3


def test_hits_at_k_parallel():
    y_true = List()
    y_pred = []
    y_t_1 = np.array([1, 4, 5, 6])
    y_p_1 = np.array([1, 2, 3, 4, 5, 7])
    y_t_2 = np.array([1, 4, 6])
    y_p_2 = np.array([1, 2, 4, 3, 5, 7])
    y_true.append(y_t_1)
    y_pred.append(y_p_1)
    y_true.append(y_t_2)
    y_pred.append(y_p_2)

    k = 5

    assert metrics.hits_at_k(y_true, np.array(y_pred), k) == 2.5


# precision_at_k ---------------------------------------------------------------
def test_precision_at_k_single():
    y_true = np.array([1, 4, 5, 6])
    y_pred = np.array([1, 2, 3, 4, 5, 7])
    k = 5
    assert metrics.precision_at_k(y_true, y_pred, k) == 3 / k


def test_precision_at_k_multi():
    y_true = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9])]
    y_pred = np.array(
        [[4, 5, 6, 2, 1, 7, 3], [4, 5, 6, 2, 1, 7, 3], [4, 5, 6, 2, 1, 7, 3]]
    )

    k = 5

    p_1 = metrics.precision_at_k(y_true[0], y_pred[0], k)
    p_2 = metrics.precision_at_k(y_true[1], y_pred[1], k)
    p_3 = metrics.precision_at_k(y_true[2], y_pred[2], k)

    mean_precision_at_k_score = sum([p_1, p_2, p_3]) / 3

    assert np.allclose(
        metrics.precision_at_k(y_true, y_pred, k), mean_precision_at_k_score
    )


def test_precision_at_k_parallel():
    y_true = List()
    y_true.append(np.array([1, 2, 3]))
    y_true.append(np.array([4, 5, 6, 7]))
    y_true.append(np.array([8, 9]))
    y_pred = np.array(
        [[4, 5, 6, 2, 1, 7, 3], [4, 5, 6, 2, 1, 7, 3], [4, 5, 6, 2, 1, 7, 3]]
    )

    k = 5

    p_1 = metrics.precision_at_k(y_true[0], y_pred[0], k)
    p_2 = metrics.precision_at_k(y_true[1], y_pred[1], k)
    p_3 = metrics.precision_at_k(y_true[2], y_pred[2], k)

    mean_precision_at_k_score = sum([p_1, p_2, p_3]) / 3

    assert np.allclose(
        metrics.precision_at_k(y_true, y_pred, k), mean_precision_at_k_score
    )


# average_precision ------------------------------------------------------------
def test_average_precision():
    y_true = [1, 4, 5, 6]
    y_pred = [1, 2, 3, 4, 5, 7]

    with pytest.raises(TypeError) as e:
        metrics.average_precision(y_true, y_pred)
        assert "y_true type not supported." in str(e.value)

    with pytest.raises(TypeError) as e:
        metrics.average_precision(np.array(y_true), y_pred)
        assert "y_true type not supported." in str(e.value)

    assert np.allclose(
        metrics.average_precision(np.array(y_true), np.array(y_pred)), 0.525
    )


# mean_average_precision -------------------------------------------------------
def test_map_multi():
    y_true = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9])]
    y_pred = np.array(
        [[4, 5, 6, 2, 1, 7, 3], [4, 5, 6, 2, 1, 7, 3], [4, 5, 6, 2, 1, 7, 3]]
    )

    ap_1 = metrics.average_precision(y_true[0], y_pred[0])
    ap_2 = metrics.average_precision(y_true[1], y_pred[1])
    ap_3 = metrics.average_precision(y_true[2], y_pred[2])

    assert np.allclose(metrics.map(y_true, y_pred), sum([ap_1, ap_2, ap_3]) / 3)


def test_map_parallel():
    y_true = List()
    y_true.append(np.array([1, 2, 3]))
    y_true.append(np.array([4, 5, 6, 7]))
    y_true.append(np.array([8, 9]))
    y_pred = np.array(
        [[4, 5, 6, 2, 1, 7, 3], [4, 5, 6, 2, 1, 7, 3], [4, 5, 6, 2, 1, 7, 3]]
    )

    ap_1 = metrics.average_precision(y_true[0], y_pred[0])
    ap_2 = metrics.average_precision(y_true[1], y_pred[1])
    ap_3 = metrics.average_precision(y_true[2], y_pred[2])

    assert np.allclose(metrics.map(y_true, y_pred), sum([ap_1, ap_2, ap_3]) / 3)


# mean_reciprocal_rank --------------------------------------------------------
def test_mrr_single():
    y_true = np.array([3])
    y_pred = np.array([2, 3, 1, 4, 5])

    assert np.allclose(metrics.mrr(y_true, y_pred), 0.5)


def test_mrr_multi():
    y_true = [np.array([3]), np.array([1]), np.array([3])]
    y_pred = np.array([[2, 3, 1, 4, 5], [1, 2, 3, 4, 5], [1, 1, 1, 4, 5]])

    assert np.allclose(metrics.mrr(y_true, y_pred), 0.5)


def test_mrr_parallel():
    y_true = List()
    y_true.append(np.array([3]))
    y_true.append(np.array([1]))
    y_true.append(np.array([3]))
    y_pred = np.array([[2, 3, 1, 4, 5], [1, 2, 3, 4, 5], [1, 1, 1, 4, 5]])

    assert np.allclose(metrics.mrr(y_true, y_pred), 0.5)


# r_precision ------------------------------------------------------------------
def test_r_precision_single():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 4, 3, 1, 5, 6, 7])

    assert np.allclose(metrics.r_precision(y_true, y_pred), 2 / 3)


def test_r_precision_multi():
    y_true = [np.array([1, 2, 3]), np.array([1, 2])]
    y_pred = np.array([[2, 4, 3, 1, 5, 6, 7], [2, 4, 3, 1, 5, 6, 7]])

    assert np.allclose(metrics.r_precision(y_true, y_pred), (2 / 3 + 1 / 2) / 2)


def test_r_precision_parallel():
    y_true = List()
    y_true.append(np.array([1, 2, 3]))
    y_true.append(np.array([1, 2]))
    y_pred = np.array([[2, 4, 3, 1, 5, 6, 7], [2, 4, 3, 1, 5, 6, 7]])

    assert np.allclose(metrics.r_precision(y_true, y_pred), (2 / 3 + 1 / 2) / 2)


# recall_at_k ------------------------------------------------------------------
def test_recall_at_k_single():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 4, 3, 1, 5, 6, 7])
    k = 2

    assert np.allclose(metrics.recall_at_k(y_true, y_pred, k), 1 / 3)


def test_recall_at_k_multi():
    y_true = [np.array([1, 2, 3]), np.array([1, 2])]
    y_pred = np.array([[2, 4, 3, 1, 5, 6, 7], [2, 4, 3, 1, 5, 6, 7]])
    k = 2

    assert np.allclose(
        metrics.recall_at_k(y_true, y_pred, k), (1 / 3 + 1 / 2) / 2
    )


def test_recall_at_k_parallel():
    y_true = List()
    y_true.append(np.array([1, 2, 3]))
    y_true.append(np.array([1, 2]))
    y_pred = np.array([[2, 4, 3, 1, 5, 6, 7], [2, 4, 3, 1, 5, 6, 7]])
    k = 2

    assert np.allclose(
        metrics.recall_at_k(y_true, y_pred, k), (1 / 3 + 1 / 2) / 2
    )


# NON-BINARY RELEVANCE METRICS =================================================
def test_dcg_single():
    # List of IDs ordered by descending order of true relevance (NOT NEEDED)
    y_true = np.array([[12, 1], [25, 1]])
    # List of IDs orderd by descending order of predicted relevance
    y_pred = np.array([12, 234, 25, 36, 32, 35])
    k = 3
    assert np.allclose(metrics.dcg(y_true, y_pred, k), 1.5)


def test_dcg_multi():
    y_true = []
    y_pred = []
    for i in range(3):
        y_true.append(np.array([[12, 1], [25, 1]]))
        y_pred.append(np.array([12, 234, 25, 36, 32, 35]))
    k = 3
    assert np.allclose(metrics.dcg(y_true, y_pred, k), 1.5)


def test_dcg_parallel():
    y_true = []
    y_pred = []
    for i in range(3):
        y_true.append(np.array([[12, 1], [25, 1]]))
        y_pred.append(np.array([12, 234, 25, 36, 32, 35]))
    k = 3
    assert np.allclose(metrics.dcg(np.array(y_true), np.array(y_pred), k), 1.5)


def test_ndcg():
    # List of IDs ordered by descending order of true relevance (NOT NEEDED)
    y_true = np.array([[2, 5], [4, 4], [5, 3], [10, 2]])
    # List of IDs orderd by descending order of predicted relevance
    y_pred_1 = np.asarray([1, 2, 3, 4, 5])  # rel = 0, 5, 0, 4, 3
    y_pred_2 = np.asarray([10, 5, 2, 4, 3])  # rel = 2, 3, 5, 4, 0
    y_pred_3 = np.asarray([1, 3, 6, 7, 8])  # rel = 0, 0, 0, 0, 0

    idcg = (
        (2 ** 5 - 1) / np.log2(2)
        + (2 ** 4 - 1) / np.log2(3)
        + (2 ** 3 - 1) / np.log2(4)
        + (2 ** 2 - 1) / np.log2(5)
    )

    k = 10

    assert np.allclose(metrics.idcg(y_true, k), idcg)

    assert np.allclose(
        metrics.ndcg(y_true, y_pred_1, k),
        (
            (2 ** 5 - 1) / np.log2(3)
            + (2 ** 4 - 1) / np.log2(5)
            + (2 ** 3 - 1) / np.log2(6)
        )
        / idcg,
    )

    assert np.allclose(
        metrics.ndcg(y_true, y_pred_2, k),
        (
            (2 ** 2 - 1) / np.log2(2)
            + (2 ** 3 - 1) / np.log2(3)
            + (2 ** 5 - 1) / np.log2(4)
            + (2 ** 4 - 1) / np.log2(5)
        )
        / idcg,
    )

    assert np.allclose(metrics.ndcg(y_true, y_pred_3, k), 0.0)


# def test_ndcg_multi():
#     # List of IDs ordered by descending order of true relevance (NOT NEEDED)
#     y_true = np.array([[2, 5], [4, 4], [5, 3], [10, 2]])
#     # List of IDs orderd by descending order of predicted relevance
#     y_pred_1 = np.asarray([1, 2, 3, 4, 5])  # rel = 0, 5, 0, 4, 3
#     y_pred_2 = np.asarray([10, 5, 2, 4, 3])  # rel = 2, 3, 5, 4, 0
#     y_pred_3 = np.asarray([1, 3, 6, 7, 8])  # rel = 0, 0, 0, 0, 0
#
#     idcg = (
#         (2 ** 5 - 1) / np.log2(2)
#         + (2 ** 4 - 1) / np.log2(3)
#         + (2 ** 3 - 1) / np.log2(4)
#         + (2 ** 2 - 1) / np.log2(5)
#     )
#
#     k = 10
#
#     ndcg_1 = (
#         (2 ** 5 - 1) / np.log2(3)
#         + (2 ** 4 - 1) / np.log2(5)
#         + (2 ** 3 - 1) / np.log2(6)
#     ) / idcg
#
#     ndcg_2 = (
#         (2 ** 2 - 1) / np.log2(2)
#         + (2 ** 3 - 1) / np.log2(3)
#         + (2 ** 5 - 1) / np.log2(4)
#         + (2 ** 4 - 1) / np.log2(5)
#     ) / idcg
#
#     ndcg_3 = 0.0
#
#     assert np.allclose(
#         metrics.ndcg(
#             [y_true, y_true, y_true],
#             np.array([y_pred_1, y_pred_2, y_pred_3]),
#             k,
#         ),
#         sum([ndcg_1, ndcg_2, ndcg_3]) / 3,
#     )
#
#
# def test_ndcg_classic():
#     # List of IDs ordered by descending order of true relevance (NOT NEEDED)
#     y_true = np.array([[2, 5], [4, 4], [5, 3], [10, 2]])
#     # List of IDs orderd by descending order of predicted relevance
#     y_pred_1 = np.asarray([1, 2, 3, 4, 5])  # rel = 0, 5, 0, 4, 3
#     y_pred_2 = np.asarray([10, 5, 2, 4, 3])  # rel = 2, 3, 5, 4, 0
#     y_pred_3 = np.asarray([1, 3, 6, 7, 8])  # rel = 0, 0, 0, 0, 0
#
#     idcg = 5 / np.log2(2) + 4 / np.log2(3) + 3 / np.log2(4) + 2 / np.log2(5)
#
#     k = 10
#
#     assert np.allclose(
#         metrics.idcg(y_true, k, method="classic"), idcg
#     )
#
#     assert np.allclose(
#         metrics.ndcg(y_true, y_pred_1, k, method="classic"),
#         (5 / np.log2(3) + 4 / np.log2(5) + 3 / np.log2(6)) / idcg,
#     )
#
#     assert np.allclose(
#         metrics.ndcg(y_true, y_pred_2, k, method="classic"),
#         (2 / np.log2(2) + 3 / np.log2(3) + 5 / np.log2(4) + 4 / np.log2(5))
#         / idcg,
#     )
#
#     assert np.allclose(
#         metrics.ndcg(y_true, y_pred_3, k, method="classic"), 0.0
#     )
#
#
# # def test_ndcg():
# #     # List of IDs ordered by descending order of true relevance (NOT NEEDED)
# #     y_true = np.array([[12, 1], [25, 1]])
# #     # List of IDs orderd by descending order of predicted relevance
# #     y_pred = np.array([12, 234, 25, 36, 32, 35])
# #     k = 3
# #     assert round(metrics.ndcg(y_true, y_pred, k), 4) == 0.9197
#
# # def test_ndcg_multi():
# #     y_true_1 = np.array([[1, 2, 3, 4, 5, 6], [5, 4, 3, 2, 1, 1]])
# #     y_pred_1 = np.array([4, 2, 1, 3, 5, 6])
# #     y_true_2 = np.array([[12, 1], [25, 1]])
# #     y_pred_2 = np.array([12, 234, 25, 36, 32, 35])
# #
# #     k = 3
# #
# #     single = (
# #         metrics.ndcg(y_true_1, y_pred_1, k)
# #         + metrics.ndcg(y_true_2, y_pred_2, k)
# #     ) / 2
# #
# #     multi = metrics.ndcg([y_true_1, y_true_2], [y_pred_1, y_pred_2], k)
# #
# #     assert single == multi
