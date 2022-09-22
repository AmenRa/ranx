from typing import List

from ..data_structures import Run
from ..fusion import fusion_switch
from ..normalization import norm_switch


def fuse(
    runs: List[Run],
    norm: str = "min-max",
    method: str = "wsum",
    params: dict = None,
):
    if params is None:
        params = {}

    # Sanity check -------------------------------------------------------------
    assert len(runs) > 1, "Only one run provided"
    for i, run_i in enumerate(runs):
        for j, run_j in enumerate(runs[i + 1 :]):
            assert (
                run_i.keys() == run_j.keys()
            ), f"Runs {i} and {j} query ids do not match"

    # Normalization ------------------------------------------------------------
    if norm is None:
        norm_runs = runs
    else:
        norm_runs = [None] * len(runs)
        if norm != "borda":
            for i, run in enumerate(runs):
                norm_runs[i] = norm_switch(norm)(run)
        else:
            norm_runs = norm_switch(norm)(runs)

    # Fusion -------------------------------------------------------------------
    return fusion_switch(method)(norm_runs, **params)
