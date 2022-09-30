from typing import List

from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Qrels, Run
from ..meta import evaluate
from .comb_sum import comb_sum
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _map_score(results, map_score):
    new_results = create_empty_results_dict()

    for i, doc_id in enumerate(results.keys()):
        new_results[doc_id] = map_score / (i + 1)

    return new_results


@njit(cache=True, parallel=True)
def _map_score_parallel(run, map_score):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _map_score(run[q_ids[i]], map_score)

    return convert_results_dict_list_to_run(q_ids, new_results)


# HIGH LEVEL FUNCTIONS =========================================================
def mapfuse(runs: List[Run], map_scores: List[float], name: str = "mapfuse"):
    r"""Computes MAPFuse as proposed by [Lillis et al.](https://dl.acm.org/doi/10.1145/1835449.1835508).

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
        runs (numba.typed.List): List of Runs.
        map_scores (float): Normalization factor.
        name (str): Name for the combined run. Defaults to "mapfuse".

    Returns:
        Fused run.

    """
    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _map_score_parallel(run.run, map_scores[i])
        _runs[i] = _run

    return comb_sum(_runs, name)


def mapfuse_train(qrels: Qrels, runs: List[Run]) -> List[float]:
    return [evaluate(qrels, run, "map") for run in runs]
