import json
from typing import Dict, List, Tuple

from tabulate import tabulate

from .frozenset_dict import FrozensetDict

chars = list("abcdefghijklmnopqrstuvwxyz")
super_chars = list("ᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ")

metric_labels = {
    "hits": "Hits",
    "precision": "P",
    "recall": "Recall",
    "r-precision": "R-Prec",
    "mrr": "MRR",
    "map": "MAP",
    "ndcg": "NDCG",
    "ndcg_burges": "NDCG_Burges",
}


class Report(object):
    def __init__(
        self,
        model_names: List[str],
        results: Dict,
        comparisons: FrozensetDict,
        metrics: List[str],
        max_p: float,
        win_tie_loss: Dict[Tuple[str], Dict[str, Dict[str, int]]],
    ):
        self.model_names = model_names
        self.results = results
        self.comparisons = comparisons
        self.metrics = metrics
        self.max_p = max_p
        self.win_tie_loss = win_tie_loss

    def get_superscript_for_table(self, model, metric):
        """Used internally."""
        return ("").join(
            [
                super_chars[j]
                for j, _model in enumerate(self.model_names)
                if model != _model
                and self.comparisons[model, _model][metric]["significant"]
                and (self.results[model][metric] > self.results[_model][metric])
            ]
        )

    def get_metric_label(self, m):
        """Used internally."""
        m_splitted = m.split("@")
        return f"{metric_labels[m_splitted[0]]}@{m_splitted[1]}"

    def to_table(self):
        """Used internally."""
        return tabulate(
            [
                [chars[i], run]
                + [
                    f"{round(score, 4)}{self.get_superscript_for_table(run, metric)}"
                    for metric, score in v.items()
                ]
                for i, (run, v) in enumerate(self.results.items())
            ],
            headers=["#", "Model"]
            + [
                self.get_metric_label(x)
                for x in list(list(self.results.values())[0].keys())
            ],
        )

    def get_superscript_for_latex(self, model, metric):
        """Used internally."""
        return ("").join(
            [
                chars[j]
                for j, _model in enumerate(self.model_names)
                if model != _model
                and self.comparisons[model, _model][metric]["significant"]
                and (self.results[model][metric] > self.results[_model][metric])
            ]
        )

    def to_latex(self):
        """Returns LaTeX table."""
        best_scores = {}

        for m in self.metrics:
            best_model = None
            best_score = 0.0
            for model in self.model_names:
                if best_score < round(self.results[model][m], 4):
                    best_score = round(self.results[model][m], 4)
                    best_model = model
            best_scores[m] = best_model

        table_prefix = (
            """========================\n% Add in preamble\n\\usepackage{graphicx}\n\setlength{\\tabcolsep}{6pt}\n========================\n\\begin{table*}[ht]\n\centering\n\caption{\nOverall effectiveness of the models.\nThe best results are highlighted in boldface.\nSuperscripts denote significant differences in Fisher's randomization test with $p\le"""
            + str(self.max_p)
            + "$.\n}\n\\resizebox{1.0\\textwidth}{!}{"
            + "\n\\begin{tabular}{c|l"
            + "|l" * len(self.metrics)
            + "}"
            + "\n\\toprule"
            + "\n\\textbf{\#}"
            + "\n& \\textbf{Model}"
            + "".join(
                [f"\n& \\textbf{{{self.get_metric_label(m)}}}" for m in self.metrics]
            )
            + " \\\\ \n\midrule"
        )

        table_content = (
            "\n".join(
                [
                    f"{chars[i]} &\n"
                    + f"{model} &\n"
                    + "\n".join(
                        [
                            "{score}$^{{{superscript}}}$ &".format(
                                score=(
                                    "\\textbf{"
                                    + f"{round(self.results[model][m], 4)}"
                                    + "}"
                                )
                                if best_scores[m] == model
                                else f"{round(self.results[model][m], 4)}",
                                superscript=self.get_superscript_for_latex(model, m),
                            )
                            for m in self.metrics
                        ]
                    )[:-1]
                    + "\\\\"
                    for i, model in enumerate(self.model_names)
                ]
            )
            .replace("_", "\\_")
            .replace("$^{}$", "")
        )

        table_suffix = (
            "\\bottomrule\n\end{tabular}\n}\n\label{tab:results}\n\end{table*}"
        )

        return table_prefix + "\n" + table_content + "\n" + table_suffix

    def print_results(self):
        print(json.dumps(self.results, indent=4))

    def __repr__(self):
        return self.to_table()

    def __str__(self):
        return self.to_table()
