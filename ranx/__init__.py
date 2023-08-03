__all__ = [
    "evaluate",
    "compare",
    "fuse",
    "normalize",
    "optimize_fusion",
    "plot",
    "Qrels",
    "Run",
]

from numba import config

from .data_structures import Qrels, Run
from .meta import compare, evaluate, fuse, normalize, optimize_fusion, plot

# Set numba threading layer to workqueue
config.THREADING_LAYER = "workqueue"
