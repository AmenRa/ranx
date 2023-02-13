from typing import List

import numpy as np
import pandas as pd
from scipy.stats import tukey_hsd


def tukey_hsd_test(
    model_names: List[str],
    scores: List[np.ndarray],
    max_p: float,
):
    """
    Performs Tukey's Honestly Significant Difference (HSD) Test.
    """
    p_values = tukey_hsd(*scores).pvalue

    res = []

    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            res.append(
                {
                    "control": model_i,
                    "treatment": model_j,
                    "p-value": p_values[i, j],
                    "significant": p_values[i, j] <= max_p,
                }
            )

    return res
