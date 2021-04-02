import numba as nb
import numpy as np
from numba.typed import List


def to_typed_list(ls):
    """Convert a list of Numpy Arrays into a Numba Typed List."""
    typed_list = List()

    for x in ls:
        typed_list.append(x)

    return typed_list
