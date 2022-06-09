from typing import List

from ..data_structures import Qrels, Run
from ..fusion.segfuse import segfuse_train


def optimize_segfuse(qrels: Qrels, runs: List[Run]) -> List[float]:
    return {"probs": segfuse_train(qrels, runs)}
