from typing import List

from rich.progress import track

from ..data_structures import Qrels, Run
from ..fusion.probfuse import estimate_probfuse_probs, probfuse
from ..meta import evaluate


def optimize_probfuse(
    qrels: Qrels,
    runs: List[Run],
    metric: str,
    min_n_segments: int = 1,
    max_n_segments: int = 100,
    show_progress: bool = True,
    return_optimization_report: bool = False,
) -> List[float]:
    trials = list(range(min_n_segments, max_n_segments + 1))

    best_score = 0.0
    best_probs = None
    optimization_report = {}

    for n_segments in track(
        trials, description="Optimizing ProbFuse", disable=not show_progress
    ):
        probs = estimate_probfuse_probs(qrels, runs, n_segments)
        fused_run = probfuse(runs, probs)
        score = evaluate(qrels, fused_run, metric, save_results_in_run=False)
        optimization_report[str(n_segments)] = score

        if score >= best_score:
            best_score = score
            best_probs = probs

    if return_optimization_report:
        return {"probs": best_probs}, optimization_report

    return {"probs": best_probs}
