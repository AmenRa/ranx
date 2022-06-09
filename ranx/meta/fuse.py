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
    if norm is not None:
        if norm != "borda":
            for i, run in enumerate(runs):
                runs[i] = norm_switch(norm)(run)
        else:
            runs = norm_switch(norm)(runs)

    # Fusion -------------------------------------------------------------------
    return fusion_switch(method)(runs, **params)
