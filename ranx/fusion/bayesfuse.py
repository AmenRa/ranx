from typing import List

import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList
from ranx.metrics import get_hit_lists

from ..data_structures import Qrels, Run
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)


@njit(cache=True)
def _sum_odds(
    results,
):
    combined_results = create_empty_results_dict()
    min_odd = np.log(0.001 / 0.999)

    for res in results:
        for doc_id in res.keys():
            if combined_results.get(doc_id, False) == False:
                combined_results[doc_id] = sum(
                    [res.get(doc_id, min_odd) for res in results]
                )

    return combined_results


@njit(cache=True, parallel=True)
def _sum_odds_parallel(runs):
    q_ids = TypedList(runs[0].keys())
    combined_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        q_id = q_ids[i]
        combined_results[i] = _sum_odds([run[q_id] for run in runs])

    return convert_results_dict_list_to_run(q_ids, combined_results)


def sum_odds(runs: List[Run]) -> Run:
    run = Run()
    run.run = _sum_odds_parallel(TypedList([run.run for run in runs]))
    run.sort()
    return run


@njit(cache=True)
def _estimate_log_odds(hit_lists, cut_offs):
    odds = np.zeros(max(cut_offs))

    for i in range(len(cut_offs) - 1):
        start = cut_offs[i]
        end = cut_offs[i + 1]

        avg_precision = 0.0
        for j in range(len(hit_lists)):
            hits = sum(hit_lists[j][start:end])
            precision = hits / (end - start)
            avg_precision += precision / len(hit_lists)

        if avg_precision == 0:
            avg_precision += 0.001

        neg_avg_precision = 1 - avg_precision

        if neg_avg_precision == 0:
            neg_avg_precision += 0.001

        odds[start:end] = [np.log(avg_precision / neg_avg_precision)] * (
            end - start
        )

    return odds


def estimate_bayesfuse_log_odds(qrels: Qrels, runs: Run) -> np.ndarray:
    cut_offs = TypedList([0, 5, 10, 15, 20, 30, 100, 200, 500, 1000])
    log_odds = []

    for run in runs:
        hit_lists = get_hit_lists(qrels.to_typed_list(), run.to_typed_list())
        log_odds.append(_estimate_log_odds(hit_lists, cut_offs))

    return log_odds


@njit(cache=True)
def _bayes_score(results, log_odds):
    new_results = create_empty_results_dict()
    run_doc_ids = TypedList(results.keys())

    for i, doc_id in enumerate(run_doc_ids):
        new_results[doc_id] = log_odds[i]

    return new_results


@njit(cache=True, parallel=True)
def _bayes_score_parallel(run, log_odds):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _bayes_score(run[q_ids[i]], log_odds)

    return convert_results_dict_list_to_run(q_ids, new_results)


def bayesfuse(
    runs: List[Run], log_odds: List[np.ndarray], name: str = "bayesfuse"
):
    r"""Computes BayesFuse as proposed by [Aslam et al.](https://dl.acm.org/doi/10.1145/383952.384007).

    ```bibtex
    @inproceedings{DBLP:conf/sigir/AslamM01,
        author    = {Javed A. Aslam and
                    Mark H. Montague},
        editor    = {W. Bruce Croft and
                    David J. Harper and
                    Donald H. Kraft and
                    Justin Zobel},
        title     = {Models for Metasearch},
        booktitle = {{SIGIR} 2001: Proceedings of the 24th Annual International {ACM} {SIGIR}
                    Conference on Research and Development in Information Retrieval, September
                    9-13, 2001, New Orleans, Louisiana, {USA}},
        pages     = {275--284},
        publisher = {{ACM}},
        year      = {2001},
        url       = {https://doi.org/10.1145/383952.384007},
        doi       = {10.1145/383952.384007},
        timestamp = {Tue, 06 Nov 2018 11:07:25 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/AslamM01.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```

    Args:
        runs (numba.typed.List): List of Runs.
        log_odds (List[np.ndarray]): Log odds of the runs' positions.
        name (str): Name for the combined run. Defaults to "bayesfuse".

    Returns:
        Run: Combined run.

    """
    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _bayes_score_parallel(run.run, log_odds[i])
        _runs[i] = _run

    run = sum_odds(_runs)
    run.name = name

    return run


def bayesfuse_train(qrels: Qrels, runs: List[Run]) -> np.ndarray:
    return estimate_bayesfuse_log_odds(qrels, runs)
