from typing import List

from ..data_structures import Qrels, Run
from ..fusion import mixed
from .optimize_weights import optimize_weights


def optimize_mixed(
    qrels: Qrels,
    runs: List[Run],
    metric: str,
    step: float = 0.1,
    show_progress: bool = True,
    return_optimization_report: bool = False,
) -> List[float]:
    return optimize_weights(
        fusion_method=mixed,
        qrels=qrels,
        runs=runs,
        metric=metric,
        step=step,
        show_progress=show_progress,
        description="Optimizing Mixed",
        return_optimization_report=return_optimization_report,
    )
