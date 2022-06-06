__all__ = [
    "average_precision",
    "f1",
    "get_hit_lists",
    "hit_rate",
    "hits",
    "ndcg_burges",
    "ndcg",
    "precision",
    "r_precision",
    "recall",
    "reciprocal_rank",
]

from .average_precision import average_precision
from .f1 import f1
from .get_hit_lists import get_hit_lists
from .hit_rate import hit_rate
from .hits import hits
from .ndcg import ndcg, ndcg_burges
from .precision import precision
from .r_precision import r_precision
from .recall import recall
from .reciprocal_rank import reciprocal_rank


def metric_switch(metric):
    if metric == "hits":
        return hits
    elif metric == "hit_rate":
        return hit_rate
    elif metric == "precision":
        return precision
    elif metric == "recall":
        return recall
    elif metric == "f1":
        return f1
    elif metric == "r-precision":
        return r_precision
    elif metric == "mrr":
        return reciprocal_rank
    elif metric == "map":
        return average_precision
    elif metric == "ndcg":
        return ndcg
    elif metric == "ndcg_burges":
        return ndcg_burges
    else:
        raise ValueError(
            f"Metric {metric} not supported. Supported metrics are `hits`, `hit_rate`, `precision`, `recall`, `f1`, `r-precision`, `mrr`, `map`, `ndcg`, and `ndcg_burges`."
        )
