from typing import List

from numba import njit, prange
from numba.typed import List as TypedList

from ..data_structures import Run
from .comb_sum import comb_sum
from .common import (
    convert_results_dict_list_to_run,
    create_empty_results_dict,
    create_empty_results_dict_list,
)


# LOW LEVEL FUNCTIONS ==========================================================
@njit(cache=True)
def _rrf_score(results, k):
    new_results = create_empty_results_dict()

    for i, doc_id in enumerate(results.keys()):
        new_results[doc_id] = 1 / (k + i + 1)

    return new_results


@njit(cache=True, parallel=True)
def _rrf_score_parallel(run, k):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _rrf_score(run[q_ids[i]], k)

    return convert_results_dict_list_to_run(q_ids, new_results)


# HIGH LEVEL FUNCTIONS =========================================================
def rrf(runs: List[Run], k: int = 60, name: str = "rrf") -> Run:
    r"""Computes Reciprocal Rank Fusion as proposed by [Cormack et al.](https://dl.acm.org/doi/10.1145/1571941.1572114).

    ```bibtex
        @inproceedings{DBLP:conf/sigir/CormackCB09,
            author    = {Gordon V. Cormack and
                        Charles L. A. Clarke and
                        Stefan B{\"{u}}ttcher},
            title     = {Reciprocal rank fusion outperforms condorcet and individual rank learning
                        methods},
            booktitle = {{SIGIR}},
            pages     = {758--759},
            publisher = {{ACM}},
            year      = {2009}
        }
    ```

    Args:
        runs (List[Run]): List of Runs.
        k (int): See the original paper.
        name (str): Name for the combined run. Defaults to "rrf".

    Returns:
        Run: Combined run.

    """
    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _rrf_score_parallel(run.run, k)
        _runs[i] = _run

    return comb_sum(_runs, name)
