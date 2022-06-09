from ..data_structures import Run
from ..normalization import norm_switch


def normalize(run: Run, norm: str = "min-max"):
    return norm_switch(norm)(run)
