__all__ = [
    "max_norm",
    "min_max_norm",
    "rank_norm",
    "sum_norm",
    "zmuv_norm",
    "borda_norm",
]

from .borda_norm import borda_norm
from .max_norm import max_norm
from .min_max_norm import min_max_norm
from .rank_norm import rank_norm
from .sum_norm import sum_norm
from .zmuv_norm import zmuv_norm


def norm_switch(method: str = "min-max"):
    if method == "borda":
        return borda_norm
    elif method == "max":
        return max_norm
    elif method in {"min_max", "min-max"}:
        return min_max_norm
    elif method == "rank":
        return rank_norm
    elif method == "sum":
        return sum_norm
    elif method == "zmuv":
        return zmuv_norm
    else:
        raise ValueError(f"Normalization method {method} not supported.")
