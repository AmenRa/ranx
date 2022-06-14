__all__ = [
    "optimize_bayesfuse",
    "optimize_comb_gmnz",
    "optimize_logn_isr",
    "optimize_mapfuse",
    "optimize_mixed",
    "optimize_posfuse",
    "optimize_probfuse",
    "optimize_rbc",
    "optimize_rrf",
    "optimize_segfuse",
    "optimize_slidefuse",
    "optimize_weighted_bordafuse",
    "optimize_weighted_condorcet",
    "optimize_wmnz",
    "optimize_wsum",
]

from .optimize_bayesfuse import optimize_bayesfuse
from .optimize_comb_gmnz import optimize_comb_gmnz
from .optimize_logn_isr import optimize_logn_isr
from .optimize_mapfuse import optimize_mapfuse
from .optimize_mixed import optimize_mixed
from .optimize_posfuse import optimize_posfuse
from .optimize_probfuse import optimize_probfuse
from .optimize_rbc import optimize_rbc
from .optimize_rrf import optimize_rrf
from .optimize_segfuse import optimize_segfuse
from .optimize_slidefuse import optimize_slidefuse
from .optimize_weighted_bordafuse import optimize_weighted_bordafuse
from .optimize_weighted_condorcet import optimize_weighted_condorcet
from .optimize_wmnz import optimize_wmnz
from .optimize_wsum import optimize_wsum


def has_hyperparams(method: str):
    if method in {
        "logn_isr",
        "gmnz",
        "mixed",
        "probfuse",
        "rbc",
        "rrf",
        "slidefuse",
        "w_bordafuse",
        "w_condorcet",
        "wmnz",
        "wsum",
    }:
        return True
    else:
        raise ValueError(f"{method} does not support optimization report.")


def optimization_switch(method: str):
    if method == "logn_isr":
        return optimize_logn_isr
    elif method == "gmnz":
        return optimize_comb_gmnz
    elif method == "mixed":
        return optimize_mixed
    elif method == "probfuse":
        return optimize_probfuse
    elif method == "rbc":
        return optimize_rbc
    elif method == "rrf":
        return optimize_rrf
    elif method == "slidefuse":
        return optimize_slidefuse
    elif method == "w_bordafuse":
        return optimize_weighted_bordafuse
    elif method == "w_condorcet":
        return optimize_weighted_condorcet
    elif method == "wmnz":
        return optimize_wmnz
    elif method == "wsum":
        return optimize_wsum
    # Training only
    elif method == "bayesfuse":
        return optimize_bayesfuse
    elif method == "mapfuse":
        return optimize_mapfuse
    elif method == "posfuse":
        return optimize_posfuse
    elif method == "segfuse":
        return optimize_segfuse
    else:
        raise ValueError(f"{method} does not support optimization.")
