import json
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

method_labels = {
    "logn_isr": "LogN_ISR",
    "gmnz": "GMNZ",
    "mixed": "Mixed",
    "probfuse": "ProbFuse",
    "rbc": "RBC",
    "rrf": "RRF",
    "slidefuse": "SlideFuse",
    "w_bordafuse": "Weighted BordaFuse",
    "w_condorcet": "Weighted Condorcet",
    "wmnz": "WMNZ",
    "wsum": "Weighted SUM",
}

hyperparams_labels = {
    "logn_isr": "Sigma",
    "gmnz": "Gamma",
    "mixed": "Weights",
    "probfuse": "# Segments",
    "rbc": "phi",
    "rrf": "k",
    "slidefuse": "w",
    "w_bordafuse": "Weights",
    "w_condorcet": "Weights",
    "wmnz": "Weights",
    "wsum": "Weights",
}

metric_labels = {
    "hits": "Hits",
    "hit_rate": "Hit_Rate",
    "precision": "P",
    "recall": "Recall",
    "f1": "F1",
    "r-precision": "R-Prec",
    "mrr": "MRR",
    "map": "MAP",
    "ndcg": "NDCG",
    "ndcg_burges": "NDCG_Burges",
}


class OptimizationReport(object):
    def __init__(
        self,
        method: str,
        configs: List[str],
        results: List[float],
        metric: List[str],
        rounding_digits: int = 3,
        show_percentages: bool = False,
    ):
        self.method = method
        self.configs = configs
        self.results = results
        self.metric = metric
        self.rounding_digits = rounding_digits
        self.show_percentages = show_percentages

    def get_metric_label(self, m):
        if "@" in m:
            m_splitted = m.split("@")
            label = metric_labels[m_splitted[0]]
            cutoff = m_splitted[1]
            return f"{label}@{cutoff}"
        return f"{metric_labels[m]}"

    def format_score(self, score):
        if self.show_percentages:
            new_score = round(score * 100, max(0, self.rounding_digits - 2))
            return "%.{n}f".format(n=self.rounding_digits - 2) % new_score
        new_score = round(score, self.rounding_digits)
        return "%.{n}f".format(n=self.rounding_digits) % new_score

    def style_best(self, x):
        return f"[bold green]{x}[/bold green]"

    def to_table(self):
        method_label = method_labels[self.method]
        table = Table(title=f"{method_label}")

        table.add_column(
            header=hyperparams_labels[self.method],
            header_style="bold magenta",
            justify="center",
        )
        table.add_column(
            header=self.get_metric_label(self.metric),
            header_style="bold magenta",
            justify="center",
        )

        max_idx = np.argmax(self.results)

        for i, (config, score) in enumerate(zip(self.configs, self.results)):
            config, score = str(config), self.format_score(score)
            if i == max_idx:
                config, score = self.style_best(config), self.style_best(score)
            table.add_row(config, score)

        return table

    def __repr__(self):
        buf = StringIO()
        console = Console(file=buf, force_jupyter=False)
        console.print(self.to_table())

        return buf.getvalue()

    def __str__(self):
        buf = StringIO()
        console = Console(file=buf, force_jupyter=False)
        console.print(self.to_table())

        return buf.getvalue()
