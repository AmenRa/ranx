from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _rank_norm(results):
    """Apply `rank norm` to a given results dictionary."""
    n_results = len(results)

    normalized_results = create_empty_results_dict()
    for i, doc_id in enumerate(results.keys()):
        normalized_results[doc_id] = 1 - (i / n_results)

    return normalized_results


@njit(cache=True, parallel=True)
def _rank_norm_parallel(run):
    """Apply `rank norm` to a each results dictionary of a run in parallel."""
    q_ids = TypedList(run.keys())

    normalized_run = create_empty_results_dict_list(len(q_ids))
    for i in prange(len(q_ids)):
        normalized_run[i] = _rank_norm(run[q_ids[i]])

    return convert_results_dict_list_to_run(q_ids, normalized_run)


# HIGH LEVEL FUNCTIONS =========================================================
def rank_norm(run):
    """Apply `rank norm` to a run."""
    normalized_run = Run()
    normalized_run.name = run.name
    normalized_run.run = _rank_norm_parallel(run.run)
    return normalized_run
