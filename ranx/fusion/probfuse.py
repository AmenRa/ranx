from math import ceil
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
def estimate_segment_probs(hit_lists, n_segments) -> np.ndarray:
    segment_probs = np.zeros(n_segments)

    for i in range(len(hit_lists)):
        hit_list = hit_lists[i]
        hit_list_len = len(hit_list)
        avg_segment_size = ceil(hit_list_len / n_segments)

        for k in range(n_segments):
            start = k * (avg_segment_size)
            end = (k + 1) * (avg_segment_size)
            segment = hit_list[start:end]
            segment_size = len(segment)
            if segment_size != 0:
                segment_probs[k] += sum(segment) / len(segment)

    segment_probs = segment_probs / len(hit_lists)

    return segment_probs


@njit(cache=True, parallel=True)
def estimate_segment_probs_multi(
    hit_lists: List, n_segments: int
) -> List[np.ndarray]:
    probs = np.zeros((len(hit_lists), n_segments))

    for i in prange(len(hit_lists)):
        probs[i] = estimate_segment_probs(hit_lists[i], n_segments)

    return probs


def estimate_probfuse_probs(qrels: Qrels, runs: List[Run], n_segments: int):
    _qrels = qrels.to_typed_list()

    # Hit lists of all the systems for all the queries
    hit_lists = TypedList(
        [get_hit_lists(_qrels, run.to_typed_list()) for run in runs]
    )

    return estimate_segment_probs_multi(hit_lists, n_segments)


@njit(cache=True)
def _prob_score(results, segment_probs):
    new_results = create_empty_results_dict()
    probs = np.array([0.0] * len(results))
    n_segments = len(segment_probs)
    segment_size = ceil(len(results) / n_segments)

    for k in prange(n_segments):
        start = k * (segment_size)
        end = (k + 1) * (segment_size)
        probs[start:end] = segment_probs[k] / (k + 1)

    for i, doc_id in enumerate(results.keys()):
        new_results[doc_id] = probs[i]

    return new_results


@njit(cache=True, parallel=True)
def _prob_score_parallel(run, probs):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _prob_score(run[q_ids[i]], probs)

    return convert_results_dict_list_to_run(q_ids, new_results)


def probfuse(runs: List[Run], probs: List[np.ndarray], name: str = "probfuse"):
    r"""Computes ProbFuse as proposed by [Lillis et al.](https://dl.acm.org/doi/10.1145/1148170.1148197).

    ```bibtex
    @inproceedings{DBLP:conf/sigir/LillisTCD06,
        author    = {David Lillis and
                    Fergus Toolan and
                    Rem W. Collier and
                    John Dunnion},
        editor    = {Efthimis N. Efthimiadis and
                    Susan T. Dumais and
                    David Hawking and
                    Kalervo J{\"{a}}rvelin},
        title     = {ProbFuse: a probabilistic approach to data fusion},
        booktitle = {{SIGIR} 2006: Proceedings of the 29th Annual International {ACM} {SIGIR}
                    Conference on Research and Development in Information Retrieval, Seattle,
                    Washington, USA, August 6-11, 2006},
        pages     = {139--146},
        publisher = {{ACM}},
        year      = {2006},
        url       = {https://doi.org/10.1145/1148170.1148197},
        doi       = {10.1145/1148170.1148197},
        timestamp = {Wed, 14 Nov 2018 10:58:10 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/LillisTCD06.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```

    Args:
        runs (List[Run]): List of Runs.
        probs (List[np.ndarray]): Probabilities for runs' positions.
        name (str, optional): Name for the combined run. Defaults to "probfuse".

    Returns:
        Run: Combined run.

    """
    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _prob_score_parallel(run.run, probs[i])
        _runs[i] = _run

    return comb_sum(_runs, name)


def probfuse_train(
    qrels: Qrels, runs: List[Run], n_segments: int
) -> List[np.ndarray]:
    return estimate_probfuse_probs(qrels, runs, n_segments)
