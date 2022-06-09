from itertools import product
from typing import Callable, List

import numpy as np
from rich.progress import track

from ..data_structures import Qrels, Run
from ..meta import evaluate


def get_possible_weights(step):
    round_digits = str(step)[::-1].find(".")
    return [round(x, round_digits) for x in np.arange(0, 1 + step, step)]


def get_trial_configs(weights, n_runs):
    return [seq for seq in product(*[weights] * n_runs) if sum(seq) == 1.0]


def optimize_weights(
    fusion_method: Callable,
    qrels: Qrels,
    runs: List[Run],
    metric: str,
    step: float = 0.1,
    show_progress: bool = True,
    description: str = "Optimizing weights",
    return_optimization_report: bool = False,
) -> List[float]:
    weights = get_possible_weights(step)
    trials = get_trial_configs(weights, len(runs))

    best_score = 0.0
    best_weights = []
    optimization_report = {}

    for weights in track(
        trials, description=description, disable=not show_progress
    ):
        fused_run = fusion_method(runs, weights)
        score = evaluate(qrels, fused_run, metric, save_results_in_run=False)
        optimization_report[str(weights)] = score

        if score >= best_score:
            best_score = score
            best_weights = weights

    if return_optimization_report:
        return {"weights": best_weights}, optimization_report

    return {"weights": best_weights}
