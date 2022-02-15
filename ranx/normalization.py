from numba.typed import Dict as TypedDict
from numba.typed import List as TypedList
from numba import types, types, prange, njit

# MAX --------------------------------------------------------------------------
@njit(cache=True)
def max_norm(results):
    normalized_results = TypedDict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    max_score = max(results.values())

    for doc_id in results.keys():
        normalized_results[doc_id] = results[doc_id] / max(max_score, 1e-9)

    return normalized_results


@njit(cache=True, parallel=True)
def max_norm_parallel(run):
    q_ids = TypedList(run.keys())

    normalized_results = TypedList(
        [
            TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.float64,
            )
            for _ in range(len(q_ids))
        ]
    )

    for i in prange(len(q_ids)):
        normalized_results[i] = max_norm(run[q_ids[i]])

    normalized_run = TypedDict()

    for i, q_id in enumerate(q_ids):
        normalized_run[q_id] = normalized_results[i]

    return normalized_run


# MIN-MAX ----------------------------------------------------------------------
@njit(cache=True)
def min_max_norm(results):
    normalized_results = TypedDict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    min_score = min(results.values())
    max_score = max(results.values())

    for doc_id in results.keys():
        normalized_results[doc_id] = (results[doc_id] - min_score) / (
            max(max_score - min_score, 1e-9)
        )

    return normalized_results


@njit(cache=True, parallel=True)
def min_max_norm_parallel(run):
    normalized_run = TypedDict()
    q_ids = TypedList(run.keys())

    normalized_results = TypedList(
        [
            TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.float64,
            )
            for _ in range(len(q_ids))
        ]
    )

    for i in prange(len(q_ids)):
        normalized_results[i] = min_max_norm(run[q_ids[i]])

    for i, q_id in enumerate(q_ids):
        normalized_run[q_id] = normalized_results[i]

    return normalized_run
