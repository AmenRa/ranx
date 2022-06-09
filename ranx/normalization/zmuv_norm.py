import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
    extract_scores,
)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _zmuv_norm(results):
    """Apply `zmuv norm` to a given results dictionary."""
    scores = extract_scores(results)
    mean_score = np.mean(scores)
    stdev_score = np.std(scores)
    denominator = max(stdev_score, 1e-9)

    normalized_results = create_empty_results_dict()
    for doc_id in results.keys():
        normalized_results[doc_id] = (results[doc_id] - mean_score) / (
            denominator
        )

    return normalized_results


@njit(cache=True, parallel=True)
def _zmuv_norm_parallel(run):
    """Apply `zmuv norm` to a each results dictionary of a run in parallel."""
    q_ids = TypedList(run.keys())

    normalized_run = create_empty_results_dict_list(len(q_ids))
    for i in prange(len(q_ids)):
        normalized_run[i] = _zmuv_norm(run[q_ids[i]])

    return convert_results_dict_list_to_run(q_ids, normalized_run)


# HIGH LEVEL FUNCTIONS =========================================================
def zmuv_norm(run):
    """Apply `zmuv norm` to a run."""
    normalized_run = Run()
    normalized_run.name = run.name
    normalized_run.run = _zmuv_norm_parallel(run.run)
    return normalized_run
