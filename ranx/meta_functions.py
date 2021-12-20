# TODO: avoid useless comparisons

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
    """Compute performance scores for all the provided metrics."""

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
