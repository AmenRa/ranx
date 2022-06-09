from typing import List

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
def _weighted_sum(results, weights):
    combined_results = create_empty_results_dict()

    for res in results:
        for doc_id in res.keys():
            if combined_results.get(doc_id, False) == False:
                combined_results[doc_id] = sum(
                    [
                        weights[i] * res.get(doc_id, 0.0)
                        for i, res in enumerate(results)
                    ]
                )

    return combined_results


@njit(cache=True, parallel=True)
def _weighted_sum_parallel(runs, weights):
    q_ids = TypedList(runs[0].keys())
    combined_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        q_id = q_ids[i]
        combined_results[i] = _weighted_sum(
            [run[q_id] for run in runs], weights
        )

    return convert_results_dict_list_to_run(q_ids, combined_results)


# HIGH LEVEL FUNCTIONS =========================================================
def wsum(
    runs: List[Run], weights: List[float], name: str = "weighted_sum"
) -> Run:
    """Computes a weighted sum of the scores given to documents by a list of Runs.

    Args:
        runs (List[Run]): List of Runs.
        weights (List[float]): Weights.
        name (str, optional): Name for the combined run. Defaults to "weighted_sum".

    Returns:
        Run: Combined run.
    """
    run = Run()
    run.name = name
    run.run = _weighted_sum_parallel(
        TypedList([run.run for run in runs]), TypedList(weights)
    )
    run.sort()
    return run
