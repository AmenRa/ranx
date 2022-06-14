__all__ = [
    "bayesfuse",
    "bayesfuse_train",
    "bordafuse",
    "comb_anz",
    "comb_gmnz",
    "comb_max",
    "comb_med",
    "comb_min",
    "comb_mnz",
    "comb_sum",
    "condorcet",
    "isr",
    "log_isr",
    "logn_isr",
    "mapfuse_train",
    "mapfuse",
    "mixed",
    "posfuse_train",
    "posfuse",
    "probfuse_train",
    "probfuse",
    "rbc",
    "rrf",
    "segfuse_train",
    "segfuse",
    "slidefuse_train",
    "slidefuse",
    "weighted_bordafuse",
    "weighted_condorcet",
    "wmnz",
    "wsum",
]

from .bayesfuse import bayesfuse, bayesfuse_train
from .bordafuse import bordafuse
from .comb_anz import comb_anz
from .comb_gmnz import comb_gmnz
from .comb_max import comb_max
from .comb_med import comb_med
from .comb_min import comb_min
from .comb_mnz import comb_mnz
from .comb_sum import comb_sum
from .condorcet import condorcet
from .isr import isr
from .log_isr import log_isr
from .logn_isr import logn_isr
from .mapfuse import mapfuse, mapfuse_train
from .mixed import mixed
from .posfuse import posfuse, posfuse_train
from .probfuse import probfuse, probfuse_train
from .rbc import rbc
from .rrf import rrf
from .segfuse import segfuse, segfuse_train
from .slidefuse import slidefuse, slidefuse_train
from .weighted_bordafuse import weighted_bordafuse
from .weighted_condorcet import weighted_condorcet
from .wmnz import wmnz
from .wsum import wsum


def fusion_switch(method):
    if method == "bayesfuse":
        return bayesfuse
    elif method == "bordafuse":
        return bordafuse
    elif method == "anz":
        return comb_anz
    elif method == "gmnz":
        return comb_gmnz
    elif method == "max":
        return comb_max
    elif method == "med":
        return comb_med
    elif method == "min":
        return comb_min
    elif method == "mnz":
        return comb_mnz
    elif method == "sum":
        return comb_sum
    elif method == "condorcet":
        return condorcet
    elif method == "isr":
        return isr
    elif method == "log_isr":
        return log_isr
    elif method == "logn_isr":
        return logn_isr
    elif method == "mapfuse":
        return mapfuse
    elif method == "mixed":
        return mixed
    elif method == "posfuse":
        return posfuse
    elif method == "probfuse":
        return probfuse
    elif method == "rbc":
        return rbc
    elif method == "rrf":
        return rrf
    elif method == "segfuse":
        return segfuse
    elif method == "slidefuse":
        return slidefuse
    elif method == "w_bordafuse":
        return weighted_bordafuse
    elif method == "w_condorcet":
        return weighted_condorcet
    elif method == "wmnz":
        return wmnz
    elif method == "wsum":
        return wsum
    else:
        raise ValueError(f"Fusion method {method} not supported.")
