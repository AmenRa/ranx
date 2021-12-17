import numpy as np
from numba import config, njit, prange
from numba.typed import Dict as TypedDict
from numba.typed import List as TypedList

config.THREADING_LAYER = "workqueue"


@njit(cache=True)
def to_typed_list(d):
    keys = TypedList(d.keys())
    values = TypedList(d.values())

    typed_list = TypedList([np.empty((1, 2), dtype=np.float64)] * len(keys))

    for i in prange(len(keys)):
        doc_ids = TypedList(values[i].keys())
        scores = TypedList(values[i].values())

        # Hash doc_ids
        doc_ids = TypedList([hash(x) for x in doc_ids])

        typed_list[i] = np.column_stack(
            (
                np.asarray(doc_ids, dtype=np.float64),
                np.asarray(scores, dtype=np.float64),
            )
        )

    return typed_list


@njit(cache=True)
def create_dict_from_lists(keys, values):
    d = TypedDict()
    for i, k in enumerate(keys):
        d[k] = values[i]
    return d


@njit(cache=True)
def typed_list_argosrt(typed_list):
    array = np.empty((len(typed_list)))
    for i in range(len(typed_list)):
        array[i] = typed_list[i]
    return np.argsort(array)


@njit(cache=True)
def sort_dict_by_key(d):
    new_d = TypedDict()

    keys = TypedList(d.keys())
    keys.sort()

    for k in keys:
        new_d[k] = d[k]

    return new_d


@njit(cache=True)
def sort_dict_by_value(d):
    new_d = TypedDict()

    keys = TypedList(d.keys())
    values = TypedList(d.values())

    for i in typed_list_argosrt(values)[::-1]:
        new_d[keys[i]] = values[i]

    return new_d


@njit(cache=True)
def sort_dict_of_dict_by_value(d):
    keys = TypedList(d.keys())
    values = TypedList(d.values()).copy()

    for i in prange(len(values)):
        values[i] = sort_dict_by_value(values[i])

    return create_dict_from_lists(keys, values)


@njit(cache=True)
def add_bulk(d, q_ids, doc_ids, scores):
    for i, q_id in enumerate(q_ids):
        d[q_id] = create_dict_from_lists(doc_ids[i], scores[i])

    return d


@njit(cache=True)
def add_and_sort(d, q_ids, doc_ids, scores):
    # Add
    d = add_bulk(d, q_ids, doc_ids, scores)
    # Sort q_ids
    d = sort_dict_by_key(d)
    # Sort scores
    d = sort_dict_of_dict_by_value(d)

    return d
