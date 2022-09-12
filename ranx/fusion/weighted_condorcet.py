from functools import cmp_to_key
from typing import List

import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)
from .condorcet import _get_run_indices, get_candidates, get_results


def _compare(x, y, run_indices, weights):
    x_votes = 0
    y_votes = 0

    for i, run in enumerate(run_indices):
        if run[x] < run[y]:
            x_votes += weights[i]
        else:
            y_votes += weights[i]

    return -1 if x_votes >= y_votes else 1


def _condorcet(results, candidates, weights):
    new_results = create_empty_results_dict()

    run_indices = _get_run_indices(results, candidates)
    sort_indices = sorted(
        list(range(len(candidates))),
        key=cmp_to_key(lambda x, y: _compare(x, y, run_indices, weights)),
    )

    doc_ids = np.array(candidates)[sort_indices]
    max_score = len(doc_ids)

    for i, doc_id in enumerate(doc_ids):
        new_results[doc_id] = max_score - i

    return new_results


def _condorcet_parallel(q_ids, results, candidates, weights):
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _condorcet(results[i], candidates[i], weights)

    return convert_results_dict_list_to_run(q_ids, new_results)


def weighted_condorcet(
    runs: List[Run], weights: List[float], name: str = "weighted_condorcet"
):
    r"""Computes Weighted CondorcetFuse as proposed by [Montague et al.](https://dl.acm.org/doi/10.1145/584792.584881).

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
        name (str): Name for the combined run. Defaults to "weighted_condorcet".

    Returns:
        Run: Combined run.

    """

    results = get_results(TypedList([run.run for run in runs]))
    candidates = get_candidates([run.run for run in runs])
    q_ids = TypedList(runs[0].keys())

    run = Run()
    run.run = _condorcet_parallel(q_ids, results, candidates, weights)
    run.name = name

    return run
