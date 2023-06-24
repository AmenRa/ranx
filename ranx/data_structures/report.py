"""`Report` stores the results of a comparison."""

import json
from typing import Dict, List, Tuple

from tabulate import tabulate

from .frozenset_dict import FrozensetDict

chars = list("abcdefghijklmnopqrstuvwxyz")
super_chars = list("ᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ")

metric_labels = {
    **{
        "hits": "Hits",
        "hit_rate": "Hit Rate",
        "precision": "P",
        "recall": "Recall",
        "f1": "F1",
        "r-precision": "R-Prec",
        "mrr": "MRR",
        "map": "MAP",
        "dcg": "DCG",
        "dcg_burges": "DCG Burges",
        "ndcg": "NDCG",
        "ndcg_burges": "NDCG Burges",
        "bpref": "BPref",
    },
    **{f"rbp.{i}": f"RBP.{i}" for i in range(1, 100)},
}

stat_test_labels = {
    "fisher": "Fisher's randomization test",
    "student": "paired Student's t-test",
    "tukey": "Tukey's HSD test",
}


class Report(object):
    """A `Report` instance is automatically generated as the results of a comparison.
    A `Report` provide a convenient way of inspecting a comparison results and exporting those il LaTeX for your scientific publications.

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
    ```python
    print(report.to_latex())  # To get the LaTeX code
    ```
    """

    def __init__(
        self,
        model_names: List[str],
        results: Dict,
        comparisons: FrozensetDict,
        metrics: List[str],
        max_p: float,
        win_tie_loss: Dict[Tuple[str], Dict[str, Dict[str, int]]],
        rounding_digits: int = 3,
        show_percentages: bool = False,
        stat_test: str = "student",
    ):
        self.model_names = model_names
        self.results = results
        self.comparisons = comparisons
        self.metrics = metrics
        self.max_p = max_p
        self.win_tie_loss = win_tie_loss
        self.rounding_digits = rounding_digits
        self.show_percentages = show_percentages
        self.stat_test = stat_test

    def format_score(self, score):
        if self.show_percentages:
            new_score = round(score * 100, max(0, self.rounding_digits - 2))
            return "%.{n}f".format(n=self.rounding_digits - 2) % new_score
        new_score = round(score, self.rounding_digits)
        return "%.{n}f".format(n=self.rounding_digits) % new_score

    def get_superscript_for_table(self, model, metric):
        superscript = [
            super_chars[j]
            for j, _model in enumerate(self.model_names)
            if model != _model
            and self.comparisons[model, _model][metric]["significant"]
            and (self.results[model][metric] > self.results[_model][metric])
        ]
        return ("").join(superscript)

    def get_metric_label(self, m):
        if "-l" in m:
            m, rel_lvl = m.split("-l")
            if "@" in m:
                m_splitted = m.split("@")
                label = metric_labels[m_splitted[0]]
                cutoff = m_splitted[1]
                return f"{label}@{cutoff}-l{rel_lvl}"
            return f"{metric_labels[m]}-l{rel_lvl}"

        else:
            if "@" in m:
                m_splitted = m.split("@")
                label = metric_labels[m_splitted[0]]
                cutoff = m_splitted[1]
                return f"{label}@{cutoff}"
            return f"{metric_labels[m]}"

    def get_stat_test_label(self, stat_test: str):
        return stat_test_labels[stat_test]

    def to_table(self):
        tabular_data = []

        for i, (run, v) in enumerate(self.results.items()):
            data = [chars[i], run]

            for metric, score in v.items():
                formatted_score = self.format_score(score)
                superscript = self.get_superscript_for_table(run, metric)
                data.append(f"{formatted_score}{superscript}")

            tabular_data.append(data)

        headers = ["#", "Model"]

        for x in self.metrics:
            label = self.get_metric_label(x)
            headers.append(label)

        return tabulate(tabular_data=tabular_data, headers=headers)

    def get_superscript_for_latex(self, model, metric):
        superscript = [
            chars[j]
            for j, _model in enumerate(self.model_names)
            if (
                model != _model
                and self.comparisons[model, _model][metric]["significant"]
                and self.results[model][metric] > self.results[_model][metric]
            )
        ]
        return ("").join(superscript)

    def get_phantoms_for_latex(self, model, metric):
        phantoms = [
            chars[j]
            for j, _model in enumerate(self.model_names)
            if (
                model != _model
                and (
                    not self.comparisons[model, _model][metric]["significant"]
                    or not self.results[model][metric] > self.results[_model][metric]
                )
            )
        ]

        if len(phantoms) > 0:
            return ("").join(phantoms)

        return ""

    def to_latex(self) -> str:
        """Returns Report as LaTeX table.

        Returns:
            str: LaTeX table
        """
        best_scores = {}

        for m in self.metrics:
            best_model = None
            best_score = 0.0
            for model in self.model_names:
                if best_score < round(self.results[model][m], self.rounding_digits):
                    best_score = round(self.results[model][m], self.rounding_digits)
                    best_model = model
            best_scores[m] = best_model

        preamble = "========================\n% Add in preamble\n\\usepackage{graphicx}\n\\usepackage{booktabs}\n========================\n\n"

        table_prefix = (
            "% To change the table size, act on the resizebox argument `0.8`.\n"
            + """\\begin{table*}[ht]\n\centering\n\caption{\nOverall effectiveness of the models.\nThe best results are highlighted in boldface.\nSuperscripts denote significant differences in """
            + self.get_stat_test_label(self.stat_test)
            + """ with $p \le """
            + str(self.max_p)
            + "$.\n}\n\\resizebox{0.8\\textwidth}{!}{"
            + "\n\\begin{tabular}{c|l"
            + "|c" * len(self.metrics)
            + "}"
            + "\n\\toprule"
            + "\n\\textbf{\#}"
            + "\n& \\textbf{Model}"
            + "".join(
                [f"\n& \\textbf{{{self.get_metric_label(m)}}}" for m in self.metrics]
            )
            + " \\\\ \n\midrule"
        )

        table_content = []

        for i, model in enumerate(self.model_names):
            table_raw = f"{chars[i]} &\n" + f"{model} &\n"
            scores = []

            for m in self.metrics:
                score = self.format_score(self.results[model][m])
                score = (
                    f"\\textbf{{{score}}}" if best_scores[m] == model else f"{score}"
                )
                superscript = self.get_superscript_for_latex(model, m)
                phantoms = self.get_phantoms_for_latex(model, m)
                scores.append(
                    f"{score}$^{{{superscript}}}$\\hphantom{{$^{{{phantoms}}}$}} &"
                )

            scores[-1] = scores[-1][:-1]  # Remove `&` at the end

            table_raw += "\n".join(scores) + "\\\\"
            table_content.append(table_raw)

        table_content = (
            "\n".join(table_content).replace("_", "\\_").replace("$^{}$", "")
        )

        table_suffix = (
            "\\bottomrule\n\end{tabular}\n}\n\label{tab:results}\n\end{table*}"
        )

        return (
            preamble + "\n" + table_prefix + "\n" + table_content + "\n" + table_suffix
        )

    def to_dict(self) -> Dict:
        """Returns the Report data as a Python dictionary.

        ```python
        {
            "stat_test": "fisher"
            # metrics and model_names allows to read the report without
            # inspecting the json to discover the used metrics and
            # the compared models
            "metrics": ["metric_1", "metric_2", ...],
            "model_names": ["model_1", "model_2", ...],
            #
            "model_1": {
                "scores": {
                    "metric_1": ...,
                    "metric_2": ...,
                    ...
                },
                "comparisons": {
                    "model_2": {
                        "metric_1": ...,  # p-value
                        "metric_2": ...,  # p-value
                        ...
                    },
                    ...
                },
                "win_tie_loss": {
                    "model_2": {
                        "W": ...,
                        "T": ...,
                        "L": ...,
                    },
                    ...
                },
            },
            ...
        }
        ```

        Returns:
            Dict: Report data as a Python dictionary
        """

        d = {
            "stat_test": self.stat_test,
            "metrics": self.metrics,
            "model_names": self.model_names,
        }

        for m1 in self.model_names:
            d[m1] = {}
            d[m1]["scores"] = self.results[m1]
            d[m1]["comparisons"] = {}
            d[m1]["win_tie_loss"] = {}

            for m2 in self.model_names:
                if m1 != m2:
                    d[m1]["comparisons"][m2] = {}
                    d[m1]["win_tie_loss"][m2] = {}

                    for metric in self.metrics:
                        d[m1]["comparisons"][m2][metric] = self.comparisons[{m1, m2}][
                            metric
                        ]["p_value"]
                        d[m1]["win_tie_loss"][m2][metric] = self.win_tie_loss[(m1, m2)][
                            metric
                        ]

        return d

    def save(self, path: str):
        """Save the Report data as JSON file.
        See [**Report.to_dict**][ranx.report.to_dict] for more details.

        Args:
            path (str): Saving path
        """
        with open(path, "w") as f:
            f.write(json.dumps(self.to_dict(), indent=4))

    def print_results(self):
        """Print report data."""
        print(json.dumps(self.results, indent=4))

    def __repr__(self):
        return self.to_table()

    def __str__(self):
        return self.to_table()
