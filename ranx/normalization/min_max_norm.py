from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
    extract_scores,
    safe_max,
    safe_min,
    to_unicode,
)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _min_max_norm(results, invert):
    """Apply `min-max norm` to a given results dictionary."""
    scores = extract_scores(results)
    min_score = safe_min(scores)
    max_score = safe_max(scores)
    denominator = max_score - min_score
    denominator = max(denominator, 1e-9)

    normalized_results = create_empty_results_dict()
    for doc_id in results.keys():
        doc_id = to_unicode(doc_id)
        if invert:
            normalized_results[doc_id] = (max_score - results[doc_id]) / (denominator)
        else:
            normalized_results[doc_id] = (results[doc_id] - min_score) / (denominator)

    return normalized_results


@njit(cache=True, parallel=True)
def _min_max_norm_parallel(run, invert):
    """Apply `min_max norm` to a each results dictionary of a run in parallel."""
    q_ids = TypedList(run.keys())

    normalized_run = create_empty_results_dict_list(len(q_ids))
    for i in prange(len(q_ids)):
        normalized_run[i] = _min_max_norm(run[q_ids[i]], invert)

    return convert_results_dict_list_to_run(q_ids, normalized_run)


# HIGH LEVEL FUNCTIONS =========================================================
def min_max_norm(run: Run, invert: bool = False) -> Run:
    """Apply `min_max norm` to a given run.

    Args:
        run (Run): Run to be normalized.

    Returns:
        Run: Normalized run.
    """
    normalized_run = Run()
    normalized_run.name = run.name
    normalized_run.run = _min_max_norm_parallel(run.run, invert)
    if invert:
        normalized_run.sort()
    return normalized_run
