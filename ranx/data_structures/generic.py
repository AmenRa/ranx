from numba import njit, types
from numba.typed import Dict as TypedDict
from numba.typed import List as TypedList


@njit(cache=True)
def create_empty_results_dict():
    return TypedDict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )


@njit(cache=True)
def create_empty_results_dict_list(length):
    return TypedList([create_empty_results_dict() for _ in range(length)])


@njit(cache=True)
def convert_results_dict_list_to_run(q_ids, results_dict_list):
    combined_run = TypedDict()

    for i, q_id in enumerate(q_ids):
        combined_run[q_id] = results_dict_list[i]

    return combined_run
