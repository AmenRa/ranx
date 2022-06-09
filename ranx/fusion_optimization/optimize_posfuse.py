from typing import List

from ..data_structures import Qrels, Run
from ..fusion.posfuse import posfuse_train


def optimize_posfuse(qrels: Qrels, runs: List[Run]) -> List[float]:
    return {"probs": posfuse_train(qrels, runs)}
