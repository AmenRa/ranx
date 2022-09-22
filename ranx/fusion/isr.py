from typing import List

from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .comb_mnz import comb_mnz
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _isr_score(results):
    new_results = create_empty_results_dict()

    for i, doc_id in enumerate(results.keys()):
        new_results[doc_id] = 1 / ((i + 1) ** 2)

    return new_results


@njit(cache=True, parallel=True)
def _isr_score_parallel(run):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _isr_score(run[q_ids[i]])

    return convert_results_dict_list_to_run(q_ids, new_results)


# HIGH LEVEL FUNCTIONS =========================================================
def isr(runs: List[Run], name: str = "isr") -> Run:
    r"""Computes ISR as proposed by [Mourão et al.](https://www.sciencedirect.com/science/article/abs/pii/S0895611114000664).

    ```bibtex
         @article{DBLP:journals/cmig/MouraoMM15,
            author    = {Andr{\'{e}} Mour{\~{a}}o and
                        Fl{\'{a}}vio Martins and
                        Jo{\~{a}}o Magalh{\~{a}}es},
            title     = {Multimodal medical information retrieval with unsupervised rank fusion},
            journal   = {Comput. Medical Imaging Graph.},
            volume    = {39},
            pages     = {35--45},
            year      = {2015},
            url       = {https://doi.org/10.1016/j.compmedimag.2014.05.006},
            doi       = {10.1016/j.compmedimag.2014.05.006},
            timestamp = {Thu, 14 May 2020 10:17:16 +0200},
            biburl    = {https://dblp.org/rec/journals/cmig/MouraoMM15.bib},
            bibsource = {dblp computer science bibliography, https://dblp.org}
        }
    ```

    Args:
        runs (List[Run]): List of Runs.
        name (str): Name for the combined run. Defaults to "isr".

    Returns:
        Run: Combined run.

    """
    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _isr_score_parallel(run.run)
        _runs[i] = _run

    run = comb_mnz(_runs)
    run.name = name

    return run
