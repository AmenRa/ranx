from typing import List

from rich.progress import track

from ..data_structures import Qrels, Run
from ..fusion.rrf import rrf
from ..meta import evaluate


def optimize_rrf(
    qrels: Qrels,
    runs: List[Run],
    metric: str,
    min_k: int = 10,
    max_k: int = 100,
    step: int = 10,
    show_progress: bool = True,
    return_optimization_report: bool = False,
) -> List[float]:
    trials = list(range(min_k, max_k + 1, step))

    best_score = 0.0
    best_k = []
    optimization_report = {}

    for k in track(
        trials, description="Optimizing RRF", disable=not show_progress
    ):
        fused_run = rrf(runs, k)
        score = evaluate(qrels, fused_run, metric, save_results_in_run=False)
        optimization_report[str(k)] = score

        if score >= best_score:
            best_score = score
            best_k = k

    if return_optimization_report:
        return {"k": best_k}, optimization_report

    return {"k": best_k}
