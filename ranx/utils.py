from collections import defaultdict
from numbers import Number
from typing import Dict

import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList
from pandas import DataFrame


@njit(cache=True)
def descending_sort(x):
    return x[np.argsort(x[:, 1])[::-1]]


@njit(cache=True, parallel=True)
def descending_sort_parallel(x):
    for i in prange(len(x)):
        x[i] = descending_sort(x[i])
    return x


# ------------------------------------------------------------------------------
def python_dict_to_typed_list(x: Dict[str, Dict[str, Number]], sort: bool = True):
    """Converts a nested Python Dictionary to Numba Typed List to be used with ranx's metrics with no effort.

    Note: Doc IDs will be hashed.
    """
    out = TypedList(
        [
            np.array(
                [[hash(doc_id), score] for doc_id, score in doc.items()],
                dtype=np.float64,
            )
            for doc in x.values()
        ]
    )

    if sort:
        out = descending_sort_parallel(out)

    return out


def qrels_file_to_dict(path: str):
    qrels_dict = defaultdict(dict)

    for x in open(path, "r").read().splitlines():
        q_id, _, doc_id, rel = x.split()
        qrels_dict[q_id][doc_id] = int(rel)

    return dict(qrels_dict)


def run_file_to_dict(path: str):
    runs_dict = defaultdict(dict)

    for x in open(path, "r").read().splitlines():
        q_id, _, doc_id, _, rel, _ = x.split()

        runs_dict[q_id][doc_id] = float(rel)

    return dict(runs_dict)


def dataframe_to_dict(df: DataFrame, q_id_col: str, doc_id_col: str, score_col: str):
    return (
        df.groupby(q_id_col)[[doc_id_col, score_col]]
        .apply(lambda g: {x[0]: x[1] for x in g.values.tolist()})
        .to_dict()
    )
