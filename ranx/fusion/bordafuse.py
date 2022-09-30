from typing import List

from numba import njit, prange
from numba.typed import List as TypedList
from numba.types import unicode_type

from ..data_structures import Run
from .comb_sum import comb_sum
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)


def get_candidates(runs):
    candidates = TypedList()
    for q_id in runs[0].keys():
        new_candidates = TypedList(
            {doc_id for run in runs for doc_id in list(run[q_id].keys())}
        )

        if len(new_candidates) > 0:
            candidates.append(new_candidates)
        else:
            # Fixes Numba raising error if no runs have docs for a given query
            candidates.append(TypedList.empty_list(unicode_type))

    return candidates


@njit(cache=True)
def _borda_score(results, candidates):
    new_results = create_empty_results_dict()
    run_doc_ids = TypedList(results.keys())
    other_doc_ids = [id for id in candidates if id not in run_doc_ids]
    max_points = len(candidates)
    unranked_points = (max_points - len(run_doc_ids) + 1) / 2

    for i, doc_id in enumerate(run_doc_ids):
        new_results[doc_id] = max_points - i

    for doc_id in other_doc_ids:
        new_results[doc_id] = unranked_points

    return new_results


@njit(cache=True, parallel=True)
def _borda_score_parallel(run, candidates):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _borda_score(run[q_ids[i]], candidates[i])

    return convert_results_dict_list_to_run(q_ids, new_results)


def bordafuse(runs: List[Run], name: str = "bordafuse"):
    r"""Computes BordaFuse as proposed by [Aslam et al.](https://dl.acm.org/doi/10.1145/383952.384007).

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
        name (str): Name for the combined run. Defaults to "bordafuse".

    Returns:
        Run: Combined run.

    """
    candidates = get_candidates(runs)

    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _borda_score_parallel(run.run, candidates)
        _runs[i] = _run

    return comb_sum(_runs, name)
