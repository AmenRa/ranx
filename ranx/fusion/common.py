from typing import List

import numpy as np
from numba import njit, prange, types
from numba.typed import Dict as TypedDict
from numba.typed import List as TypedList

from ..data_structures import Qrels, Run
from ..metrics import get_hit_lists


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


@njit(cache=True)
def _estimate_probs(hit_lists):
    hit_list_lens = [len(hit_list) for hit_list in hit_lists]

    denominators = np.zeros(max(hit_list_lens))
    numerators = np.zeros(max(hit_list_lens))

    for hit_list in hit_lists:
        for i in prange(len(hit_list)):
            denominators[i] += 1
            numerators[i] += hit_list[i]

    return numerators / denominators


def estimate_probs(qrels: Qrels, run: Run) -> np.ndarray:
    hit_lists = get_hit_lists(qrels.to_typed_list(), run.to_typed_list())
    return _estimate_probs(hit_lists)


def estimate_probs_multi(qrels: Qrels, runs: List[Run]) -> List[np.ndarray]:
    return [estimate_probs(qrels, run) for run in runs]


@njit(cache=True)
def extract_scores(results):
    """Extract the scores from a given results dictionary."""
    scores = np.empty((len(results)))
    for i, v in enumerate(results.values()):
        scores[i] = v
    return scores
