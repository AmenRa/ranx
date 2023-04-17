from collections import defaultdict
from typing import List, Union

import numpy as np

from ..data_structures import FrozensetDict, Qrels, Report, Run
from ..statistical_tests import compute_statistical_significance
from .evaluate import evaluate, format_metrics


def compare(
    qrels: Qrels,
    runs: List[Run],
    metrics: Union[List[str], str],
    stat_test: str = "student",
    n_permutations: int = 1000,
    max_p: float = 0.01,
    random_seed: int = 42,
    threads: int = 0,
    rounding_digits: int = 3,
    show_percentages: bool = False,
    make_comparable: bool = False,
) -> Report:
    """Evaluate multiple `runs` and compute statistical tests.

    Usage example:
    ```python
    from ranx import compare

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
        stat_test (str, optional): Statistical test to perform. Use "fisher" for _Fisher's Randomization Test_, "student" for _Two-sided Paired Student's t-Test_, or "Tukey" for _Tukey's HSD test_. Defaults to "student".
        random_seed (int, optional): Random seed to use for generating the permutations. Defaults to 42.
        threads (int, optional): Number of threads to use, zero means all the available threads. Defaults to 0.
        rounding_digits (int, optional): Number of digits to round to and to show in the Report. Defaults to 3.
        show_percentages (bool, optional): Whether to show percentages instead of floats in the Report. Defaults to False.
        make_comparable (bool, optional): Adds empty results for queries missing from the runs and removes those not appearing in qrels. Defaults to False.

    Returns:
        Report: See report.
    """
    metrics = format_metrics(metrics)
    assert all(type(m) == str for m in metrics), "Metrics error"

    model_names = []
    results = defaultdict(dict)

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
            make_comparable=make_comparable,
        )

        if len(metrics) == 1:
            metric_scores[model_name] = {metrics[0]: metric_scores[model_name]}

        for m in metrics:
            results[model_name][m] = float(
                np.mean(metric_scores[model_name][m])
            )

    # Run statistical testing --------------------------------------------------
    comparisons = compute_statistical_significance(
        model_names=model_names,
        metric_scores=metric_scores,
        stat_test=stat_test,
        n_permutations=n_permutations,
        max_p=max_p,
        random_seed=random_seed,
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
        stat_test=stat_test,
    )
