__all__ = [
    "fisher_randomization_test",
    "paired_student_t_test",
    "compute_statistical_significance",
]

from .fisher_randomization_test import fisher_randomization_test
from .paired_student_t_test import paired_student_t_test


def compute_statistical_significance(
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
