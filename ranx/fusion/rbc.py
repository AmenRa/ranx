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


@njit(cache=True)
def _rbc_score(results, phi):
    new_results = create_empty_results_dict()
    run_doc_ids = TypedList(results.keys())

    for i, doc_id in enumerate(run_doc_ids):
        new_results[doc_id] = (1 - phi) * phi**i

    return new_results


@njit(cache=True, parallel=True)
def _rbc_score_parallel(run, phi):
    q_ids = TypedList(run.keys())
    new_results = create_empty_results_dict_list(len(q_ids))

    for i in prange(len(q_ids)):
        new_results[i] = _rbc_score(run[q_ids[i]], phi)

    return convert_results_dict_list_to_run(q_ids, new_results)


def rbc(runs: List[Run], phi: float, name: str = "rbc"):
    r"""Computes Rank-Biased Centroid (RBC) as proposed by [Bailey et al.](https://dl.acm.org/doi/10.1145/3077136.3080839).

    ```bibtex
    @inproceedings{DBLP:conf/sigir/BaileyMST17,
        author    = {Peter Bailey and
                    Alistair Moffat and
                    Falk Scholer and
                    Paul Thomas},
        editor    = {Noriko Kando and
                    Tetsuya Sakai and
                    Hideo Joho and
                    Hang Li and
                    Arjen P. de Vries and
                    Ryen W. White},
        title     = {Retrieval Consistency in the Presence of Query Variations},
        booktitle = {Proceedings of the 40th International {ACM} {SIGIR} Conference on
                    Research and Development in Information Retrieval, Shinjuku, Tokyo,
                    Japan, August 7-11, 2017},
        pages     = {395--404},
        publisher = {{ACM}},
        year      = {2017},
        url       = {https://doi.org/10.1145/3077136.3080839},
        doi       = {10.1145/3077136.3080839},
        timestamp = {Wed, 25 Sep 2019 16:43:14 +0200},
        biburl    = {https://dblp.org/rec/conf/sigir/BaileyMST17.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```

    Args:
        runs (numba.typed.List): List of Runs.
        phi (float): Persistence / patience parameter.
        name (str): Name for the combined run. Defaults to "rbc".

    Returns:
        Run: Combined run.

    """
    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _rbc_score_parallel(run.run, phi)
        _runs[i] = _run

    return comb_sum(_runs, name)
