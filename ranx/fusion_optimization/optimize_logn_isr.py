from typing import List

import numpy as np
from rich.progress import track

from ..data_structures import Qrels, Run
from ..fusion.logn_isr import logn_isr
from ..meta import evaluate


def optimize_logn_isr(
    qrels: Qrels,
    runs: List[Run],
    metric: str,
    min_sigma: float = 0.01,
    max_sigma: float = 1.0,
    step: float = 0.01,
    show_progress: bool = True,
    return_optimization_report: bool = False,
) -> List[float]:
    rounding_digits = str(step)[::-1].find(".")
    trials = [
        round(x, rounding_digits)
        for x in np.arange(min_sigma, max_sigma + step, step)
    ]

    best_score = 0.0
    best_sigma = []
    optimization_report = {}

    for sigma in track(
        trials, description="Optimizing LogN_ISR", disable=not show_progress
    ):
        fused_run = logn_isr(runs, sigma)
        score = evaluate(qrels, fused_run, metric, save_results_in_run=False)
        optimization_report[str(sigma)] = score

        if score >= best_score:
            best_score = score
            best_sigma = sigma

    if return_optimization_report:
        return {"sigma": best_sigma}, optimization_report

    return {"sigma": best_sigma}
