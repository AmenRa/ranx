from typing import List

from ..data_structures import Qrels, Run
from ..fusion.bayesfuse import bayesfuse_train


def optimize_bayesfuse(qrels: Qrels, runs: List[Run],) -> List[float]:
    return {"log_odds": bayesfuse_train(qrels, runs)}
