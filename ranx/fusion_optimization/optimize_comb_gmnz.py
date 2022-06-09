from typing import List

import numpy as np
from rich.progress import track

from ..data_structures import Qrels, Run
from ..fusion.comb_gmnz import comb_gmnz
from ..meta import evaluate


def optimize_comb_gmnz(
    qrels: Qrels,
    runs: List[Run],
    metric: str,
    min_gamma: float = 0.01,
    max_gamma: float = 1.0,
    step: float = 0.01,
    show_progress: bool = True,
    return_optimization_report: bool = False,
) -> List[float]:
    rounding_digits = str(step)[::-1].find(".")
    trials = [
        round(x, rounding_digits)
        for x in np.arange(min_gamma, max_gamma + step, step)
    ]

    best_score = 0.0
    best_gamma = []
    optimization_report = {}

    for gamma in track(
        trials, description="Optimizing CombGMNZ", disable=not show_progress
    ):
        fused_run = comb_gmnz(runs, gamma)
        score = evaluate(qrels, fused_run, metric, save_results_in_run=False)
        optimization_report[str(gamma)] = score

        if score >= best_score:
            best_score = score
            best_gamma = gamma

    if return_optimization_report:
        return {"gamma": best_gamma}, optimization_report

    return {"gamma": best_gamma}
