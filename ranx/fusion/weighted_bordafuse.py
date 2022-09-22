from typing import List

from numba.typed import List as TypedList

from ..data_structures import Run
from .bordafuse import _borda_score_parallel, get_candidates
from .wsum import wsum


def weighted_bordafuse(
    runs: List[Run], weights: List[float], name: str = "weighted_bordafuse"
):
    r"""Computes Weighted BordaFuse as proposed by [Aslam et al.](https://dl.acm.org/doi/10.1145/383952.384007).

    ```bibtex
    @inproceedings{DBLP:conf/sigir/AslamM01,
        author    = {Javed A. Aslam and
                    Mark H. Montague},
        editor    = {W. Bruce Croft and
                    David J. Harper and
                    Donald H. Kraft and
                    Justin Zobel},
        title     = {Models for Metasearch},
        booktitle = {{SIGIR} 2001: Proceedings of the 24th Annual International {ACM} {SIGIR}
                    Conference on Research and Development in Information Retrieval, September
                    9-13, 2001, New Orleans, Louisiana, {USA}},
        pages     = {275--284},
        publisher = {{ACM}},
        year      = {2001},
        url       = {https://doi.org/10.1145/383952.384007},
        doi       = {10.1145/383952.384007},
        timestamp = {Tue, 06 Nov 2018 11:07:25 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/AslamM01.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```

    Args:
        runs (numba.typed.List): List of Runs.
        weights (List[float]): Weights.
        name (str): Name for the combined run. Defaults to "weighted_bordafuse".

    Returns:
        Run: Combined run.

    """
    candidates = get_candidates(runs)

    _runs = [None] * len(runs)
    for i, run in enumerate(runs):
        _run = Run()
        _run.run = _borda_score_parallel(run.run, candidates)
        _runs[i] = _run

    run = wsum(_runs, weights)
    run.name = name

    return run
