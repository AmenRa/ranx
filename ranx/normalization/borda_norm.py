from typing import List

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


@njit(cache=True)
def _borda_norm(results, candidates):
    doc_ids = TypedList(results.keys())
    n_results = len(results)
    n_candidates = len(candidates)

    normalized_results = create_empty_results_dict()
    for doc_id in candidates:
        if doc_id in results:
            normalized_results[doc_id] = 1 - (
                doc_ids.index(doc_id) / n_candidates
            )
        else:
            normalized_results[doc_id] = 0.5 - (
                (n_results - 1) / (2 * n_candidates)
            )

    return normalized_results


@njit(cache=True, parallel=True)
def _borda_norm_parallel(run, candidates):
    q_ids = TypedList(run.keys())

    normalized_run = create_empty_results_dict_list(len(q_ids))
    for i in prange(len(q_ids)):
        normalized_run[i] = _borda_norm(run[q_ids[i]], candidates[i])

    return convert_results_dict_list_to_run(q_ids, normalized_run)


def borda_norm(runs: List[Run]):
    """Apply `borda norm` to a list of runs."""
    candidates = get_candidates(runs)

    normalized_runs = []

    for run in runs:
        normalized_run = Run()
        normalized_run.name = run.name
        normalized_run.run = _borda_norm_parallel(run.run, candidates)
        normalized_runs.append(normalized_run)

    return normalized_runs
