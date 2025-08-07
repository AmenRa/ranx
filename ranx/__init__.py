__all__ = [
    "evaluate",
    "compare",
    "fuse",
    "normalize",
    "optimize_fusion",
    "plot",
    "Qrels",
    "Run",
    "use_numba",
    "set_numba_enabled",
]

from .config import set_numba_enabled, use_numba
from .data_structures import Qrels, Run
from .meta import compare, evaluate, fuse, normalize, optimize_fusion, plot

# Conditional Numba configuration
if use_numba():
    try:
        from numba import config

        # Set numba threading layer to workqueue
        config.THREADING_LAYER = "workqueue"
    except ImportError:
        # Numba not available, silently continue
        pass
