from functools import cmp_to_key
from typing import List

import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList
from numba.types import unicode_type

from ..data_structures import Run
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


# def get_candidates(runs):
#     candidates = TypedList()
#     for q_id in runs[0]:
#         new_candidates = np.array(
#             list({doc_id for run in runs for doc_id in list(run[q_id].keys())})
#         )

#         if len(new_candidates) > 0:
#             candidates.append(new_candidates)
#         else:
#             # Fixes Numba raising error if no runs have docs for a given query
#             candidates.append(np.array(["str"])[:0])

#     return candidates

# def get_candidates(runs):
#     candidates = TypedList()
#     for q_id in runs[0]:
#         candidates.append(
#             np.array(
#                 list(
#                     {
#                         doc_id
#                         for run in runs
#                         for doc_id in list(run[q_id].keys())
#                     }
#                 )
#             )
#         )

#     return candidates


@njit(cache=True)
def get_results(runs):
    results = TypedList()
    for q_id in runs[0].keys():
        q_results = TypedList()
        for run in runs:
            q_results.append(TypedList(run[q_id].keys()))
        results.append(q_results)

    return results


@njit(cache=True)
def _get_run_indices(results, candidates):
    run_indices = TypedList()

    for run in results:
        _run_indices = TypedList()
        for x in candidates:
            try:
                _run_indices.append(run.index(str(x)))
            except:
                _run_indices.append(1_000_000)

        run_indices.append(_run_indices)

    return run_indices


def _compare(x, y, run_indices):
    x_votes = sum(run[x] < run[y] for run in run_indices)
    y_votes = len(run_indices) - x_votes
    return -1 if x_votes >= y_votes else 1


def _condorcet(results, candidates):
    new_results = create_empty_results_dict()

    run_indices = _get_run_indices(results, candidates)
    sort_indices = sorted(
        list(range(len(candidates))),
        key=cmp_to_key(lambda x, y: _compare(x, y, run_indices)),
    )

    doc_ids = np.array(candidates)[sort_indices]
    max_score = len(doc_ids)

    for i, doc_id in enumerate(doc_ids):
        new_results[doc_id] = max_score - i

    return new_results


def _condorcet_parallel(q_ids, results, candidates):
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _condorcet(results[i], candidates[i])

    return convert_results_dict_list_to_run(q_ids, new_results)


def condorcet(runs: List[Run], name: str = "condorcet"):
    r"""Computes CondorcetFuse as proposed by [Montague et al.](https://dl.acm.org/doi/10.1145/584792.584881).

    ```bibtex
    @inproceedings{DBLP:conf/cikm/MontagueA02,
        author    = {Mark H. Montague and
                    Javed A. Aslam},
        title     = {Condorcet fusion for improved retrieval},
        booktitle = {Proceedings of the 2002 {ACM} {CIKM} International Conference on Information
                    and Knowledge Management, McLean, VA, USA, November 4-9, 2002},
        pages     = {538--548},
        publisher = {{ACM}},
        year      = {2002},
        url       = {https://doi.org/10.1145/584792.584881},
        doi       = {10.1145/584792.584881},
        timestamp = {Tue, 06 Nov 2018 16:57:50 +0100},
        biburl    = {https://dblp.org/rec/conf/cikm/MontagueA02.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```

    Args:
        runs (numba.typed.List): List of Runs.
        name (str): Name for the combined run. Defaults to "condorcet".

    Returns:
        Run: Combined run.

    """
    results = get_results(TypedList([run.run for run in runs]))
    candidates = get_candidates([run.run for run in runs])
    q_ids = TypedList(runs[0].keys())

    run = Run()
    run.run = _condorcet_parallel(q_ids, results, candidates)
    run.name = name

    return run
