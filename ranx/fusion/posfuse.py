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
def _pos_score(results, probs):
    new_results = create_empty_results_dict()
    len_probs = len(probs)

    for i, doc_id in enumerate(results.keys()):
        new_results[doc_id] = probs[i] if i < len_probs else 0.0

    return new_results


@njit(cache=True, parallel=True)
def _pos_score_parallel(run, probs):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _pos_score(run[q_ids[i]], probs)

    return convert_results_dict_list_to_run(q_ids, new_results)


def posfuse(
    runs: List[Run], probs: List[np.ndarray], name: str = "posfuse"
) -> Run:
    r"""Computes PosFuse as proposed by [Lillis et al.](https://dl.acm.org/doi/10.1145/1835449.1835508).

    ```bibtex
    @inproceedings{DBLP:conf/sigir/LillisZTCLD10,
        author    = {David Lillis and
                    Lusheng Zhang and
                    Fergus Toolan and
                    Rem W. Collier and
                    David Leonard and
                    John Dunnion},
        editor    = {Fabio Crestani and
                    St{\'{e}}phane Marchand{-}Maillet and
                    Hsin{-}Hsi Chen and
                    Efthimis N. Efthimiadis and
                    Jacques Savoy},
        title     = {Estimating probabilities for effective data fusion},
        booktitle = {Proceeding of the 33rd International {ACM} {SIGIR} Conference on Research
                    and Development in Information Retrieval, {SIGIR} 2010, Geneva, Switzerland,
                    July 19-23, 2010},
        pages     = {347--354},
        publisher = {{ACM}},
        year      = {2010},
        url       = {https://doi.org/10.1145/1835449.1835508},
        doi       = {10.1145/1835449.1835508},
        timestamp = {Tue, 06 Nov 2018 11:07:25 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/LillisZTCLD10.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```

    Args:
        runs (List[Run]): List of Runs.
        probs (List[np.ndarray]): Probabilities for runs' positions.
        name (str, optional): Name for the combined run. Defaults to "posfuse".

    Returns:
        Run: Combined run.
    """
    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _pos_score_parallel(run.run, probs[i])
        _runs[i] = _run

    return comb_sum(_runs, name)


def posfuse_train(qrels: Qrels, runs: List[Run]) -> List[np.ndarray]:
    return estimate_probs_multi(qrels, runs)
