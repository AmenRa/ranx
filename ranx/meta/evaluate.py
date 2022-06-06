from numbers import Number
from typing import Dict, List, Union

import numba as nb
import numpy as np
from numba import set_num_threads

from ..data_structures import Qrels, Run
from ..metrics import metric_switch
from ..utils import python_dict_to_typed_list


def format_metrics(metrics: Union[List[str], str]) -> List[str]:
    if type(metrics) == str:
        metrics = [metrics]
    return metrics


def format_k(k: Union[List[int], int], metrics: List[str]) -> Dict[str, int]:
    if type(k) == int:
        return {m: k for m in metrics}
    elif type(k) == list:
        return {m: k[i] for i, m in enumerate(list(metrics))}
    return k


def extract_metric_and_k(metric):
    metric_splitted = metric.split("@")
    m = metric_splitted[0]

    k = int(metric_splitted[1]) if len(metric_splitted) > 1 else 0

    return m, k


def convert_qrels(qrels):
    if type(qrels) == Qrels:
        return qrels.to_typed_list()
    elif type(qrels) == dict:
        return python_dict_to_typed_list(qrels, sort=True)
    return qrels


def convert_run(run):
    if type(run) == Run:
        return run.to_typed_list()
    elif type(run) == dict:
        return python_dict_to_typed_list(run, sort=True)
    return run


def check_keys(qrels, run):
    assert qrels.keys() == run.keys(), "Qrels and Run query ids do not match"


def evaluate(
    qrels: Union[
        Qrels,
        Dict[str, Dict[str, Number]],
        nb.typed.typedlist.List,
        np.ndarray,
    ],
    run: Union[
        Run, Dict[str, Dict[str, Number]], nb.typed.typedlist.List, np.ndarray,
    ],
    metrics: Union[List[str], str],
    return_mean: bool = True,
    threads: int = 0,
    save_results_in_run=True,
) -> Union[Dict[str, float], float]:
    """Compute the performance scores for the provided `qrels` and `run` for all the specified metrics.

    Usage examples:

    ```python
    from ranx import evaluate
    
    # Compute score for a single metric
    evaluate(qrels, run, "ndcg@5")
    >>> 0.7861

    # Compute scores for multiple metrics at once
    evaluate(qrels, run, ["map@5", "mrr"])
    >>> {"map@5": 0.6416, "mrr": 0.75}

    # Computed metric scores are saved in the Run object
    run.mean_scores
    >>> {"ndcg@5": 0.7861, "map@5": 0.6416, "mrr": 0.75}

    # Access scores for each query
    dict(run.scores)
    >>> {"ndcg@5": {"q_1": 0.9430, "q_2": 0.6292},
          "map@5": {"q_1": 0.8333, "q_2": 0.4500},
            "mrr": {"q_1": 1.0000, "q_2": 0.5000}}
    ```

    Args:
        qrels (Union[ Qrels, Dict[str, Dict[str, Number]], nb.typed.typedlist.List, np.ndarray, ]): Qrels.
        run (Union[ Run, Dict[str, Dict[str, Number]], nb.typed.typedlist.List, np.ndarray, ]): Run.
        metrics (Union[List[str], str]): Metrics or list of metric to compute.
        return_mean (bool, optional): Wether to return the metric scores averaged over the query set or the scores for individual queries. Defaults to True.
        threads (int, optional): Number of threads to use, zero means all the available threads. Defaults to 0.
        save_results_in_run (bool, optional): Save metric scores for each query in the input `run`. Defaults to True.

    Returns:
        Union[Dict[str, float], float]: Results.
    """

    if len(qrels) < 10:
        set_num_threads(1)
    elif threads != 0:
        set_num_threads(threads)

    if type(qrels) in [Qrels, dict] and type(run) in [Run, dict]:
        check_keys(qrels, run)

    _qrels = convert_qrels(qrels)
    _run = convert_run(run)
    metrics = format_metrics(metrics)
    assert all(type(m) == str for m in metrics), "Metrics error"

    # Compute metrics ----------------------------------------------------------
    metric_scores_dict = {}
    for metric in metrics:
        m, k = extract_metric_and_k(metric)
        metric_scores_dict[metric] = metric_switch(m)(_qrels, _run, k=k,)

    # Save results in Run ------------------------------------------------------
    if type(run) == Run and save_results_in_run:
        for m, scores in metric_scores_dict.items():
            run.mean_scores[m] = np.mean(scores)
            for i, q_id in enumerate(run.get_query_ids()):
                run.scores[m][q_id] = scores[i]

    # Prepare output -----------------------------------------------------------
    if return_mean:
        for m, scores in metric_scores_dict.items():
            metric_scores_dict[m] = np.mean(scores)
    if len(metrics) == 1:
        return metric_scores_dict[m]

    return metric_scores_dict
