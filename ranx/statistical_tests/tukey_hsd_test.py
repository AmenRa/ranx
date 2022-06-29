from typing import List

import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def tukey_hsd_test(
    model_names: List[str], scores: List[np.ndarray], max_p: float,
):
    """
    Performs Tukey's Honestly Significant Difference (HSD) Test.
    """
    n_queries = len(scores[0])

    results = pairwise_tukeyhsd(
        endog=np.concatenate(scores),
        groups=np.array(
            [x for name in model_names for x in [name] * n_queries]
        ),
        alpha=max_p,
    )

    # Convert results to Pandas DataFrame
    df = pd.DataFrame(
        data=np.array(results._results_table.data[1:])[:, [0, 1, 3, -1]],
        columns=["control", "treatment", "p-value", "significant"],
    )

    # Extract data as Python Dictionaries
    return df.to_dict(orient="records")
