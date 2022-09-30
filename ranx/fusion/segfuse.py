from typing import List

import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Qrels, Run
from ..metrics import get_hit_lists
from .comb_sum import comb_sum
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)


@njit(cache=True)
def estimate_segment_probs(hit_lists) -> np.ndarray:
    segment_sizes = [0] + [(10 * 2 ** (k - 1)) - 5 for k in range(1, 11)]
    segment_probs = np.array([0.0] * (len(segment_sizes) - 1))

    for hit_list in hit_lists:
        for k in prange(len(segment_sizes) - 1):
            start = segment_sizes[k]
            end = segment_sizes[k + 1]
            segment_size = end - start
            segment_probs[k] += sum(hit_list[start:end]) / segment_size

    segment_probs = segment_probs / len(hit_lists)

    return segment_probs


def estimate_segment_probs_multi(hit_lists: List) -> List[np.ndarray]:
    return [estimate_segment_probs(_hit_lists) for _hit_lists in hit_lists]


def estimate_segfuse_probs(qrels: Qrels, runs: List[Run]):
    _qrels = qrels.to_typed_list()

    # Hit lists of all the systems for all the queries
    hit_lists = TypedList(
        [get_hit_lists(_qrels, run.to_typed_list()) for run in runs]
    )

    return estimate_segment_probs_multi(hit_lists)


@njit(cache=True)
def _seg_score(results, segment_probs):
    new_results = create_empty_results_dict()
    probs = np.array([0.0] * len(results))
    segment_sizes = [0] + [(10 * 2 ** (k - 1)) - 5 for k in range(1, 11)]

    for k in prange(len(segment_probs)):
        start = segment_sizes[k]
        end = segment_sizes[k + 1]
        probs[start:end] = segment_probs[k]

    for i, doc_id in enumerate(results.keys()):
        new_results[doc_id] = probs[i] * (results[doc_id] + 1)

    return new_results


@njit(cache=True, parallel=True)
def _seg_score_parallel(run, probs):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _seg_score(run[q_ids[i]], probs)

    return convert_results_dict_list_to_run(q_ids, new_results)


def segfuse(runs: List[Run], probs: List[np.ndarray], name: str = "segfuse"):
    r"""Computes SegFuse as proposed by [Shokouhi](https://link.springer.com/chapter/10.1007/978-3-540-78646-7_33).

    ```bibtex
        @inproceedings{DBLP:conf/ecir/Shokouhi07a,
            author    = {Milad Shokouhi},
            editor    = {Giambattista Amati and
                        Claudio Carpineto and
                        Giovanni Romano},
            title     = {Segmentation of Search Engine Results for Effective Data-Fusion},
            booktitle = {Advances in Information Retrieval, 29th European Conference on {IR}
                        Research, {ECIR} 2007, Rome, Italy, April 2-5, 2007, Proceedings},
            series    = {Lecture Notes in Computer Science},
            volume    = {4425},
            pages     = {185--197},
            publisher = {Springer},
            year      = {2007},
            url       = {https://doi.org/10.1007/978-3-540-71496-5\_19},
            doi       = {10.1007/978-3-540-71496-5\_19},
            timestamp = {Tue, 14 May 2019 10:00:37 +0200},
            biburl    = {https://dblp.org/rec/conf/ecir/Shokouhi07a.bib},
            bibsource = {dblp computer science bibliography, https://dblp.org}
        }
    ```

    Args:
        runs (numba.typed.List): List of Runs.
        probs (List[np.ndarray]): Probabilities for runs' positions.
        name (str): Name for the combined run. Defaults to "segfuse".

    Returns:
        Run: Combined run.

    """
    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _seg_score_parallel(run.run, probs[i])
        _runs[i] = _run

    return comb_sum(_runs, name)


def segfuse_train(qrels: Qrels, runs: List[Run]) -> List[np.ndarray]:
    return estimate_segfuse_probs(qrels, runs)
