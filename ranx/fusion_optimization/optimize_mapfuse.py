from typing import List

from ..data_structures import Qrels, Run
from ..fusion.mapfuse import mapfuse_train


def optimize_mapfuse(qrels: Qrels, runs: List[Run]) -> List[float]:
    return {"map_scores": mapfuse_train(qrels, runs)}
