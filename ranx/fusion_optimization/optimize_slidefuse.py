from typing import List

from rich.progress import track

from ..data_structures import Qrels, Run
from ..fusion.slidefuse import slidefuse, slidefuse_train
from ..meta import evaluate


def optimize_slidefuse(
    qrels: Qrels,
    runs: List[Run],
    metric: str,
    min_w: int = 1,
    max_w: int = 100,
    show_progress: bool = True,
    return_optimization_report: bool = False,
) -> List[float]:
    trials = list(range(min_w, max_w + 1))

    best_score = 0.0
    best_w = []
    optimization_report = {}

    probs = slidefuse_train(qrels, runs)

    for w in track(
        trials, description="Optimizing SlideFuse", disable=not show_progress
    ):
        fused_run = slidefuse(runs, probs, w)
        score = evaluate(qrels, fused_run, metric, save_results_in_run=False)
        optimization_report[str(w)] = score

        if score >= best_score:
            best_score = score
            best_w = w

    if return_optimization_report:
        return {"probs": probs, "w": best_w}, optimization_report

    return {"probs": probs, "w": best_w}
