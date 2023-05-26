import yaml

from .data_structures import Run
from .io import save_lz4

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
        "ndcg": "NDCG",
        "ndcg_burges": "NDCG Burges",
        "bpref": "BPref",
    },
    **{f"rbp.{i}": f"RBP.{i}" for i in range(1, 100)},
}


def get_metric_label(m):
    if "-l" in m:
        m, _ = m.split("-l")
    if "@" in m:
        m_splitted = m.split("@")
        label = metric_labels[m_splitted[0]]
        cutoff = m_splitted[1]
        return f"{label}@{cutoff}"
    return f"{metric_labels[m]}"


def save(run: Run, runcard_path: str, path: str) -> None:
    with open(runcard_path, "r") as f:
        runcard = yaml.load(f, Loader=yaml.Loader)

    for metric, score in run.mean_scores.items():
        label = get_metric_label(metric)
        runcard["run"]["results"] = runcard["run"].get("results", {})
        runcard["run"]["results"][label] = float(score)

    content = {"metadata": runcard, "run": run.to_dict()}

    with open(runcard_path, "w") as f:
        runcard = f.write(yaml.dump(runcard, sort_keys=False))

    save_lz4(content, path)
