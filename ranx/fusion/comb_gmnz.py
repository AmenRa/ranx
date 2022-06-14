from typing import List

import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .common import (convert_results_dict_list_to_run,
                     create_empty_results_dict, create_empty_results_dict_list)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _comb_gmnz(results, gamma):
    combined_results = create_empty_results_dict()

    for res in results:
        for doc_id in res.keys():
            if combined_results.get(doc_id, False) == False:
                scores = np.array(
                    [res[doc_id] for res in results if doc_id in res]
                )
                combined_results[doc_id] = sum(scores) * (len(scores) ** gamma)

    return combined_results


@njit(cache=True, parallel=True)
def _comb_gmnz_parallel(runs, gamma):
    q_ids = TypedList(runs[0].keys())
    combined_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        q_id = q_ids[i]
        combined_results[i] = _comb_gmnz([run[q_id] for run in runs], gamma)

    return convert_results_dict_list_to_run(q_ids, combined_results)


# HIGH LEVEL FUNCTIONS =========================================================
def comb_gmnz(runs: List[Run], gamma: float, name: str = "comb_gmnz") -> Run:
    r"""Computes CombGMNZ as proposed by [Joon Ho Lee](https://dl.acm.org/doi/10.1145/258525.258587).

    ```bibtex
        @inproceedings{DBLP:conf/sigir/Lee97,
            author    = {Joon Ho Lee},
            title     = {Analyses of Multiple Evidence Combination},
            booktitle = {{SIGIR}},
            pages     = {267--276},
            publisher = {{ACM}},
            year      = {1997}
        }
    ```

    Args:
        runs (List[Run]): List of Runs.
        gamma (float): Gamma parameter.
        name (str): Name for the combined run. Defaults to "comb_gmnz".

    Returns:
        Run: Combined run.

    """
    run = Run()
    run.name = name
    run.run = _comb_gmnz_parallel(TypedList([run.run for run in runs]), gamma)
    run.sort()
    return run
