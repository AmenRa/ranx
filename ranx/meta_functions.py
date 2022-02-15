from collections import defaultdict
from itertools import product
from numbers import Number
from typing import Dict, List, Union

import numba as nb
import numpy as np
from numba import set_num_threads
from numba.typed import List as TypedList
from tqdm import tqdm

from .frozenset_dict import FrozensetDict
from .fusion import weighted_sum
from .metrics import (
    average_precision,
    f1,
    hit_rate,
    hits,
    ndcg,
    ndcg_burges,
    precision,
    r_precision,
    recall,
    reciprocal_rank,
)
from .normalization import max_norm_parallel, min_max_norm_parallel
from .qrels import Qrels
from .report import Report
from .run import Run
from .statistical_testing import fisher_randomization_test
from .utils import python_dict_to_typed_list


def metric_functions_switch(metric):
    if metric == "hits":
        return hits
    elif metric == "hit_rate":
        return hit_rate
    elif metric == "precision":
        return precision
    elif metric == "recall":
        return recall
    elif metric == "f1":
        return f1
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
            f"Metric {metric} not supported. Supported metrics are `hits`, `hit_rate`, `precision`, `recall`, `f1`, `r-precision`, `mrr`, `map`, `ndcg`, and `ndcg_burges`."
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
    """Used internally."""
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
    rounding_digits: int = 3,
    show_percentages: bool = False,
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
    for i, run in enumerate(runs):
        model_name = run.name if run.name is not None else f"run_{i+1}"
        model_names.append(model_name)

        metric_scores[model_name] = evaluate(
            qrels=qrels,
            run=run,
            metrics=metrics,
            return_mean=False,
            threads=threads,
        )

        if len(metrics) == 1:
            metric_scores[model_name] = {metrics[0]: metric_scores[model_name]}

        for m in metrics:
            results[model_name][m] = float(
                np.mean(metric_scores[model_name][m])
            )

    # Run statistical testing --------------------------------------------------
    for control in model_names:
        control_metric_scores = metric_scores[control]
        for treatment in model_names:
            if control != treatment:
                treatment_metric_scores = metric_scores[treatment]

                # Compute statistical significance
                comparisons[
                    frozenset([control, treatment])
                ] = compute_statistical_significance(
                    control_metric_scores,
                    treatment_metric_scores,
                    n_permutations,
                    max_p,
                    random_seed,
                )

    # Compute win / tie / lose -------------------------------------------------
    win_tie_loss = defaultdict(dict)

    for control in model_names:
        for treatment in model_names:
            if control != treatment:
                for m in metrics:
                    control_scores = metric_scores[control][m]
                    treatment_scores = metric_scores[treatment][m]
                    win_tie_loss[(control, treatment)][m] = {
                        "W": int(sum(control_scores > treatment_scores)),
                        "T": int(sum(control_scores == treatment_scores)),
                        "L": int(sum(control_scores < treatment_scores)),
                    }

    return Report(
        model_names=model_names,
        results=dict(results),
        comparisons=comparisons,
        metrics=metrics,
        max_p=max_p,
        win_tie_loss=dict(win_tie_loss),
        rounding_digits=rounding_digits,
        show_percentages=show_percentages,
    )


# FUSION -----------------------------------------------------------------------
def fuse(
    runs: List[Run],
    kind: str = "wsum",
    params: dict = None,
    norm: str = "max",
    name: str = "fused_run",
):
    """Disclaimer: THIS IS AN EXPERIMENTAL FEATURE!
    Fuses a list of runs using the specified function and parameters.
    Only score weighted sum is currently supported.
    Params must be `{weights: weight_list}`

    Args:
        runs (List[Run]): List of runs to fuse.
        kind (str, optional): Fusion function. Only weighted sum is currently supported. Defaults to "wsum".
        params (dict, optional): Parameters for the fusion function. Defaults to None.
        norm (str, optional): Normalization to apply before fusion. Defaults to "max".
        name (str, optional): Name of the fused run. Defaults to "fused_run".

    Returns:
        Run: fused run.
        best_params:
    """
    assert len(runs) > 1, "Only one run provided"
    assert len(runs) <= 10, "Too many runs provided"
    assert all(
        runs[0].keys() == run.keys() for run in runs
    ), "Runs query ids do not match"

    if params is None:
        params = {}

    # Extract Numba Typed Dict
    runs = TypedList([run.run.copy() for run in runs])

    # Scores normalization -----------------------------------------------------
    if norm == "max":
        for i, run in enumerate(runs):
            runs[i] = max_norm_parallel(run)
    elif norm == "min_max":
        for i, run in enumerate(runs):
            runs[i] = min_max_norm_parallel(run)
    else:
        raise NotImplementedError()

    # Fusion -------------------------------------------------------------------
    if kind in {"wsum", "weighted_sum"}:
        if "weights" not in params:
            params["weights"] = [0.5 for _ in runs]
        run = weighted_sum(runs, np.array(params["weights"]))
    else:
        raise NotImplementedError()

    fused_run = Run()
    fused_run.name = name
    fused_run.run = run

    return fused_run


def optimize_fusion(
    qrels: Qrels,
    runs: List[Run],
    kind: str = "wsum",
    norm: str = "max",
    name: str = "fused_run",
    search_kind: str = "greed",
    optimize_metric: str = None,
    optimize_kwargs: dict = None,
):
    """Disclaimer: THIS IS AN EXPERIMENTAL FEATURE!"""
    if search_kind != "greed":
        raise NotImplementedError()

    assert len(runs) > 1, "Only one run provided"
    assert len(runs) <= 10, "Too many runs provided"
    assert all(
        runs[0].keys() == run.keys() for run in runs
    ), "Runs query ids do not match"

    # Extract Numba Typed Dict
    runs = TypedList([run.run.copy() for run in runs])

    # Scores normalization -----------------------------------------------------
    if norm == "max":
        for i, run in enumerate(runs):
            runs[i] = max_norm_parallel(run)
    elif norm == "min_max":
        for i, run in enumerate(runs):
            runs[i] = min_max_norm_parallel(run)
    else:
        raise NotImplementedError()

    # Fusion -------------------------------------------------------------------
    if kind in {"wsum", "weighted_sum"}:
        step = optimize_kwargs["step"]
        rounding_digits = str(step)[::-1].find(".")
        weights = [
            round(x, rounding_digits) for x in np.arange(0, 1 + step, step)
        ]

        if optimize_kwargs.get("kind", False) == "convex":
            trials = [
                seq
                for seq in product(*[weights] * len(runs))
                if sum(seq) == 1.0
            ]
        else:
            trials = list(product(*[weights for _ in runs]))

        best_score = 0.0
        best_weights = []

        for weights in tqdm(
            trials,
            desc="Optimizing fusion",
            position=0,
            dynamic_ncols=True,
            mininterval=0.5,
        ):
            run = Run()
            run.run = weighted_sum(runs, np.array(weights))
            score = evaluate(
                qrels, run, optimize_metric, save_results_in_run=False
            )

            if score > best_score:
                best_score = score
                best_weights = weights

        run = weighted_sum(runs, np.array(best_weights))
        best_params = {"weights": best_weights}

    else:
        raise NotImplementedError()

    fused_run = Run()
    fused_run.name = name
    fused_run.run = run

    return fused_run, best_params, best_score
