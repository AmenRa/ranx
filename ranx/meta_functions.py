from collections import defaultdict
from numbers import Number
from typing import Dict, List, Union

import numba as nb
import numpy as np
from numba import set_num_threads

from .frozenset_dict import FrozensetDict
from .metrics import (
    average_precision,
    hits,
    ndcg,
    ndcg_burges,
    precision,
    r_precision,
    recall,
    reciprocal_rank,
)
from .qrels import Qrels
from .report import Report
from .run import Run
from .statistical_testing import fisher_randomization_test
from .utils import python_dict_to_typed_list


def metric_functions_switch(metric):
    if metric == "hits":
        return hits
    elif metric == "precision":
        return precision
    elif metric == "recall":
        return recall
    elif metric == "r-precision":
        return r_precision
    elif metric == "mrr":
        return reciprocal_rank
    elif metric == "map":
        return average_precision
    elif metric == "ndcg":
        return ndcg
    elif metric == "ndcg_burges":
        return ndcg_burges
    else:
        raise ValueError(
            f"Metric {metric} not supported. Supported metrics are `hits`, `precision`, `recall`, `r-precision`, `mrr`, `map`, `ndcg`, and `ndcg_burges`."
        )


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
        Run,
        Dict[str, Dict[str, Number]],
        nb.typed.typedlist.List,
        np.ndarray,
    ],
    metrics: Union[List[str], str],
    return_mean: bool = True,
    threads: int = 0,
    save_results_in_run=True,
) -> Union[Dict[str, float], float]:
    """Compute the performance scores for the provided `qrels` and `run` for all the specified metrics.

    Usage examples:

    ```python
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
        metric_scores_dict[metric] = metric_functions_switch(m)(
            _qrels,
            _run,
            k=k,
        )

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


def compute_statistical_significance(
    control_metric_scores,
    treatment_metric_scores,
    n_permutations: int = 1000,
    max_p: float = 0.01,
    random_seed: int = 42,
):
    metric_p_values = {}

    for m in list(control_metric_scores):
        (
            control_mean,
            treatment_mean,
            p_value,
            significant,
        ) = fisher_randomization_test(
            control_metric_scores[m],
            treatment_metric_scores[m],
            n_permutations,
            max_p,
            random_seed,
        )

        metric_p_values[m] = {
            "p_value": p_value,
            "significant": significant,
        }

    return metric_p_values


def compare(
    qrels: Qrels,
    runs: List[Run],
    metrics: Union[List[str], str],
    n_permutations: int = 1000,
    max_p: float = 0.01,
    random_seed: int = 42,
    threads: int = 0,
):
    """Evaluate multiple `runs` and compute statistical tests.

    Usage example:
    ```python
    # Compare different runs and perform statistical tests
    report = compare(
        qrels=qrels,
        runs=[run_1, run_2, run_3, run_4, run_5],
        metrics=["map@100", "mrr@100", "ndcg@10"],
        max_p=0.01  # P-value threshold
    )

    print(report)
    ```
    Output:
    ```
    #    Model    MAP@100     MRR@100     NDCG@10
    ---  -------  ----------  ----------  ----------
    a    model_1  0.3202ᵇ     0.3207ᵇ     0.3684ᵇᶜ
    b    model_2  0.2332      0.2339      0.239
    c    model_3  0.3082ᵇ     0.3089ᵇ     0.3295ᵇ
    d    model_4  0.3664ᵃᵇᶜ   0.3668ᵃᵇᶜ   0.4078ᵃᵇᶜ
    e    model_5  0.4053ᵃᵇᶜᵈ  0.4061ᵃᵇᶜᵈ  0.4512ᵃᵇᶜᵈ
    ```

    Args:
        qrels (Qrels): Qrels.
        runs (List[Run]): List of runs.
        metrics (Union[List[str], str]): Metric or list of metrics.
        n_permutations (int, optional): Number of permutation to perform during statistical testing (Fisher's Randomization Test is used by default). Defaults to 1000.
        max_p (float, optional): Maximum p-value to consider an increment as statistically significant. Defaults to 0.01.
        random_seed (int, optional): Random seed to use for generating the permutations. Defaults to 42.
        threads (int, optional): Number of threads to use, zero means all the available threads. Defaults to 0.

    Returns:
        Report: See report.
    """
    metrics = format_metrics(metrics)
    assert all(type(m) == str for m in metrics), "Metrics error"

    model_names = []
    results = defaultdict(dict)
    comparisons = FrozensetDict()

    metric_scores = {}

    # Compute scores for each run for each query -------------------------------
    for run in runs:
        model_names.append(run.name)
        metric_scores[run.name] = evaluate(
            qrels=qrels,
            run=run,
            metrics=metrics,
            return_mean=False,
            threads=threads,
        )
        for m in metrics:
            results[run.name][m] = np.mean(metric_scores[run.name][m])

    # Run statistical testing --------------------------------------------------
    for i, control in enumerate(runs):
        control_metric_scores = metric_scores[control.name]
        for j, treatment in enumerate(runs):
            if i < j:
                treatment_metric_scores = metric_scores[treatment.name]

                # Compute statistical significance
                comparisons[
                    frozenset([control.name, treatment.name])
                ] = compute_statistical_significance(
                    control_metric_scores,
                    treatment_metric_scores,
                    n_permutations,
                    max_p,
                    random_seed,
                )

    # Compute win / tie / lose -------------------------------------------------
    win_tie_loss = defaultdict(dict)

    for control in runs:
        for treatment in runs:
            for m in metrics:
                control_scores = metric_scores[control.name][m]
                treatment_scores = metric_scores[treatment.name][m]
                win_tie_loss[(control.name, treatment.name)][m] = {
                    "W": sum(control_scores > treatment_scores),
                    "T": sum(control_scores == treatment_scores),
                    "L": sum(control_scores < treatment_scores),
                }

    return Report(
        model_names=model_names,
        results=dict(results),
        comparisons=comparisons,
        metrics=metrics,
        max_p=max_p,
        win_tie_loss=dict(win_tie_loss),
    )
