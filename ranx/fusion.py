from numba.typed import Dict as TypedDict
from numba.typed import List as TypedList
from numba import types, njit, types, prange

# WEIGHTED SUM -----------------------------------------------------------------
@njit(cache=True)
def weighted_sum_combine_query_results(results, weights):
    combined_results = TypedDict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

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
def weighted_sum(runs, weights):
    q_ids = TypedList(runs[0].keys())

    combined_results = TypedList(
        [
            TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.float64,
            )
            for _ in range(len(q_ids))
        ]
    )

    for i in prange(len(q_ids)):
        q_id = q_ids[i]
        combined_results[i] = weighted_sum_combine_query_results(
            [run[q_id] for run in runs], weights
        )

    combined_run = TypedDict()

    for i, q_id in enumerate(q_ids):
        combined_run[q_id] = combined_results[i]

    return combined_run
