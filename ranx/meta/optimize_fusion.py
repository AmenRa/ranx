from typing import List

from ..data_structures import Qrels, Run
from ..fusion_optimization import optimization_switch
from ..normalization import norm_switch


def optimize_fusion(
    qrels: Qrels,
    runs: List[Run],
    norm: str = "min-max",
    method: str = "wsum",
    metric: str = "ndcg",
    return_optimization_report: bool = False,
    **kwargs,
):
    if kwargs is None:
        kwargs = {}

    if method in {
        "gmnz",
        "logn_isr",
        "mixed",
        "probfuse",
        "rrf",
        "slidefuse",
        "w_bordafuse",
        "w_condorcet",
        "wmnz",
        "wsum",
    }:
        kwargs["metric"] = metric
        if return_optimization_report:
            kwargs["return_optimization_report"] = return_optimization_report

    # Sanity check -------------------------------------------------------------
    assert len(runs) > 1, "Only one run provided"
    for i, run_i in enumerate(runs):
        for j, run_j in enumerate(runs[i + 1 :]):
            assert (
                run_i.keys() == run_j.keys()
            ), f"Runs {i} and {j} query ids do not match"
    for i, run in enumerate(runs):
        assert (
            run.keys() == qrels.keys()
        ), f"Run {i} and Qrels query ids do not match"

    # Normalization ------------------------------------------------------------
    if norm is not None:
        if norm != "borda":
            for i, run in enumerate(runs):
                runs[i] = norm_switch(norm)(run)
        else:
            runs = norm_switch(norm)(runs)

    # Optimize fusion ----------------------------------------------------------
    return optimization_switch(method)(qrels, runs, **kwargs)
