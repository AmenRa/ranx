from typing import List

import numpy as np
from rich.progress import track

from ..data_structures import Qrels, Run
from ..fusion.rbc import rbc
from ..meta import evaluate


def optimize_rbc(
    qrels: Qrels,
    runs: List[Run],
    metric: str,
    min_phi: float = 0.01,
    max_phi: float = 1.0,
    step: float = 0.01,
    show_progress: bool = True,
    return_optimization_report: bool = False,
) -> List[float]:
    rounding_digits = str(step)[::-1].find(".")
    trials = [
        round(x, rounding_digits)
        for x in np.arange(min_phi, max_phi + step, step)
    ]

    best_score = 0.0
    best_phi = []
    optimization_report = {}

    for phi in track(
        trials, description="Optimizing RBC", disable=not show_progress
    ):
        fused_run = rbc(runs, phi)
        score = evaluate(qrels, fused_run, metric, save_results_in_run=False)
        optimization_report[str(phi)] = score

        if score >= best_score:
            best_score = score
            best_phi = phi

    if return_optimization_report:
        return {"phi": best_phi}, optimization_report

    return {"phi": best_phi}
