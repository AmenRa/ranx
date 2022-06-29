__all__ = [
    "fisher_randomization_test",
    "paired_student_t_test",
    "compute_statistical_significance",
]

from typing import Dict, List

import numpy as np

from ..data_structures import FrozensetDict
from .fisher_randomization_test import fisher_randomization_test
from .paired_student_t_test import paired_student_t_test
from .tukey_hsd_test import tukey_hsd_test


def _compute_statistical_significance(
    control_metric_scores,
    treatment_metric_scores,
    stat_test: str = "fisher",
    n_permutations: int = 1000,
    max_p: float = 0.01,
    random_seed: int = 42,
):
    """Used internally."""
    metric_p_values = {}

    for m in list(control_metric_scores):
        if stat_test == "fisher":
            p_value, significant = fisher_randomization_test(
                control_metric_scores[m],
                treatment_metric_scores[m],
                n_permutations,
                max_p,
                random_seed,
            )

        elif stat_test == "student":
            p_value, significant = paired_student_t_test(
                control_metric_scores[m], treatment_metric_scores[m], max_p,
            )

        else:
            raise NotImplementedError(
                f"Statistical test `{stat_test}` not supported."
            )

        metric_p_values[m] = {
            "p_value": p_value,
            "significant": significant,
        }

    return metric_p_values


def compute_statistical_significance(
    model_names: List[str],
    metric_scores: Dict[str, Dict[str, np.ndarray]],
    stat_test: str = "fisher",
    n_permutations: int = 1000,
    max_p: float = 0.01,
    random_seed: int = 42,
):
    """Used internally."""
    comparisons = FrozensetDict()

    if stat_test in {"fisher", "student"}:
        for control in model_names:
            control_metric_scores = metric_scores[control]
            for treatment in model_names:
                if control != treatment:
                    treatment_metric_scores = metric_scores[treatment]

                    # Compute statistical significance
                    comparisons[
                        frozenset([control, treatment])
                    ] = _compute_statistical_significance(
                        control_metric_scores,
                        treatment_metric_scores,
                        stat_test,
                        n_permutations,
                        max_p,
                        random_seed,
                    )

        return comparisons
    elif stat_test in {"tukey"}:
        metrics = list(metric_scores[model_names[0]])

        # Initialize comparisons
        for i, control in enumerate(model_names):
            for treatment in model_names[i:]:
                comparisons[frozenset([control, treatment])] = {
                    m: None for m in metrics
                }

        for m in metrics:
            scores = [metric_scores[name][m] for name in model_names]
            results = tukey_hsd_test(
                model_names=model_names, scores=scores, max_p=max_p,
            )

            for res in results:
                comparisons[res["control"], res["treatment"]][m] = {
                    "p_value": res["p-value"],
                    "significant": res["significant"],
                }

    else:
        raise NotImplementedError(
            f"Statistical test `{stat_test}` not supported."
        )

    return comparisons
