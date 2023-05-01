from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..data_structures import Qrels, Run
from ..metrics import interpolated_precision_at_recall


def plot(
    qrels: Qrels,
    runs: Union[Run, List[Run]],
    graph: str = "iprec_at_recall",
    return_graph: bool = True,
    figsize: tuple = None,
    seaborn_kwargs: Dict = None,
):
    if figsize is None:
        figsize = (8, 5)

    if seaborn_kwargs is None:
        seaborn_kwargs = {"linewidth": 2.5}

    _qrels = qrels.to_typed_list()

    if type(runs) == list:
        _runs = [run.to_typed_list() for run in runs]
        names = [
            run.name if run.name is not None else f"run_{i+1}"
            for i, run in enumerate(runs)
        ]
    else:
        _runs = [runs.to_typed_list()]
        names = [runs.name if runs.name is not None else "run"]

    results = [
        interpolated_precision_at_recall(_qrels, run).mean(axis=0)
        for run in _runs
    ]

    recall = np.arange(0, 1.1, 0.1)

    df = pd.DataFrame(
        {
            "Model": [name for name in names for _ in range(11)],
            "recall": np.concatenate([recall] * len(_runs)),
            "precision": np.concatenate(results),
        }
    )

    if not return_graph:
        return df

    plt.figure(figsize=figsize)

    ax = sns.lineplot(
        data=df, x="recall", y="precision", hue="Model", **seaborn_kwargs
    )

    ax.set(
        title="Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xticks=np.arange(0.0, 1.1, 0.1),
        yticks=np.arange(0.0, 1.1, 0.1),
    )

    plt.show()
