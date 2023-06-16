__all__ = [
    "average_precision",
    "bpref",
    "f1",
    "get_hit_lists",
    "hit_rate",
    "hits",
    "dcg_burges",
    "dcg",
    "ndcg_burges",
    "ndcg",
    "precision",
    "r_precision",
    "rank_biased_precision",
    "recall",
    "reciprocal_rank",
    "interpolated_precision_at_recall",
]

from .average_precision import average_precision
from .bpref import bpref
from .f1 import f1
from .get_hit_lists import get_hit_lists
from .hit_rate import hit_rate
from .hits import hits
from .interpolated_precision_at_recall import interpolated_precision_at_recall
from .ndcg import dcg, dcg_burges, ndcg, ndcg_burges
from .precision import precision
from .r_precision import r_precision
from .rank_biased_precision import rank_biased_precision
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
    elif metric == "dcg":
        return dcg
    elif metric == "dcg_burges":
        return dcg_burges
    elif metric == "ndcg":
        return ndcg
    elif metric == "ndcg_burges":
        return ndcg_burges
    elif metric == "bpref":
        return bpref
    elif metric == "rbp":
        return rank_biased_precision
    else:
        raise ValueError(
            f"Metric {metric} not supported. Supported metrics are `hits`, `hit_rate`, `precision`, `recall`, `f1`, `r-precision`, `mrr`, `map`, `ndcg`, and `ndcg_burges`."
        )
