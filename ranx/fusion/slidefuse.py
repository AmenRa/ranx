from typing import List

import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Qrels, Run
from .comb_sum import comb_sum
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
    estimate_probs_multi,
)


@njit(cache=True)
def _slide_score(results, probs, w):
    new_results = create_empty_results_dict()
    N = len(results)

    for p, doc_id in enumerate(results.keys()):
        a = max(p - w, 0)
        b = min(p + w, N - 1)
        # We add 1 to b because of NumPy slicing
        new_results[doc_id] = sum(probs[a : b + 1]) / (b - a + 1)

    return new_results


@njit(cache=True, parallel=True)
def _slide_score_parallel(run, probs, w):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _slide_score(run[q_ids[i]], probs, w)

    return convert_results_dict_list_to_run(q_ids, new_results)


def slidefuse(
    runs: List[Run],
    probs: List[np.ndarray],
    w: int,
    name: str = "slidefuse",
) -> Run:
    r"""Computes SlideFuse as proposed by [Lillis et al.](https://link.springer.com/chapter/10.1007/978-3-540-78646-7_33).

    ```bibtex
    @inproceedings{DBLP:conf/ecir/LillisTCD08,
        author    = {David Lillis and
                    Fergus Toolan and
                    Rem W. Collier and
                    John Dunnion},
        editor    = {Craig Macdonald and
                    Iadh Ounis and
                    Vassilis Plachouras and
                    Ian Ruthven and
                    Ryen W. White},
        title     = {Extending Probabilistic Data Fusion Using Sliding Windows},
        booktitle = {Advances in Information Retrieval , 30th European Conference on {IR}
                    Research, {ECIR} 2008, Glasgow, UK, March 30-April 3, 2008. Proceedings},
        series    = {Lecture Notes in Computer Science},
        volume    = {4956},
        pages     = {358--369},
        publisher = {Springer},
        year      = {2008},
        url       = {https://doi.org/10.1007/978-3-540-78646-7\_33},
        doi       = {10.1007/978-3-540-78646-7\_33},
        timestamp = {Sun, 25 Oct 2020 22:33:08 +0100},
        biburl    = {https://dblp.org/rec/conf/ecir/LillisTCD08.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```

    Args:
        runs (numba.typed.List): List of Runs.
        probs (List[np.ndarray]): Probabilities for runs' positions.
        w (int): Sliding window size.
        name (str): Name for the combined run. Defaults to "posfuse".

    Returns:
        Run: Combined run.

    """
    _runs = [None] * len(runs)

    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _slide_score_parallel(run.run, probs[i], w)
        _runs[i] = _run

    return comb_sum(_runs, name)


def slidefuse_train(qrels: Qrels, runs: List[Run]) -> List[np.ndarray]:
    return estimate_probs_multi(qrels, runs)
