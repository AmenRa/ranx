from typing import List

import numpy as np
from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _wmnz(results, weights):
    combined_results = create_empty_results_dict()

    for res in results:
        for doc_id in res.keys():
            if combined_results.get(doc_id, False) == False:
                scores_and_weights = np.array(
                    [
                        [res[doc_id], weights[i]]
                        for i, res in enumerate(results)
                        if doc_id in res
                    ]
                )

                scores_sum = sum(scores_and_weights[:, 0])
                weights_sum = sum(scores_and_weights[:, 1])

                combined_results[doc_id] = scores_sum * weights_sum

    return combined_results


@njit(cache=True, parallel=True)
def _wmnz_parallel(runs, weights):
    q_ids = TypedList(runs[0].keys())
    combined_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        q_id = q_ids[i]
        combined_results[i] = _wmnz([run[q_id] for run in runs], weights)

    return convert_results_dict_list_to_run(q_ids, combined_results)


# HIGH LEVEL FUNCTIONS =========================================================
def wmnz(runs: List[Run], weights: List[float], name: str = "wmnz") -> Run:
    r"""Computes Weighted MNZ as proposed by [Wu et al.](https://dl.acm.org/doi/10.1145/584792.584908).

    ```bibtex
        @inproceedings{DBLP:conf/cikm/WuC02,
            author    = {Shengli Wu and
                        Fabio Crestani},
            title     = {Data fusion with estimated weights},
            booktitle = {Proceedings of the 2002 {ACM} {CIKM} International Conference on Information
                        and Knowledge Management, McLean, VA, USA, November 4-9, 2002},
            pages     = {648--651},
            publisher = {{ACM}},
            year      = {2002},
            url       = {https://doi.org/10.1145/584792.584908},
            doi       = {10.1145/584792.584908},
            timestamp = {Tue, 06 Nov 2018 16:57:40 +0100},
            biburl    = {https://dblp.org/rec/conf/cikm/WuC02.bib},
            bibsource = {dblp computer science bibliography, https://dblp.org}
        }
    ```

    Args:
        runs (List[Run]): List of Runs.
        weights (List[float]): Weights.
        name (str): Name for the combined run. Defaults to "wmnz".

    Returns:
        Run: Combined run.

    """
    run = Run()
    run.name = name
    run.run = _wmnz_parallel(
        TypedList([run.run for run in runs]), TypedList(weights)
    )
    run.sort()
    return run
