from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
    extract_scores,
    safe_min,
)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _sum_norm(results):
    """Apply `sum norm` to a given results dictionary."""
    scores = extract_scores(results)
    min_score = safe_min(scores)
    sum_score = sum(scores)
    denominator = sum_score - min_score * len(results)
    denominator = max(denominator, 1e-9)

    normalized_results = create_empty_results_dict()
    for doc_id in results.keys():
        normalized_results[doc_id] = (results[doc_id] - min_score) / (
            denominator
        )

    return normalized_results


@njit(cache=True, parallel=True)
def _sum_norm_parallel(run):
    """Apply `sum norm` to a each results dictionary of a run in parallel."""
    q_ids = TypedList(run.keys())

    normalized_run = create_empty_results_dict_list(len(q_ids))
    for i in prange(len(q_ids)):
        normalized_run[i] = _sum_norm(run[q_ids[i]])

    return convert_results_dict_list_to_run(q_ids, normalized_run)


# HIGH LEVEL FUNCTIONS =========================================================
def sum_norm(run):
    """Apply `sum norm` to a run."""
    normalized_run = Run()
    normalized_run.name = run.name
    normalized_run.run = _sum_norm_parallel(run.run)
    return normalized_run
