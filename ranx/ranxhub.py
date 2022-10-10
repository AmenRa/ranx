import yaml

from ranx.data_structures.qrels import Qrels

from .data_structures import Run
from .io import save_lz4
from .meta import evaluate


def save(qrels: Qrels, run: Run, runcard_path: str, path: str) -> None:
    metrics = evaluate(
        qrels=qrels,
        run=run,
        metrics=[
            "hit_rate",
            "hit_rate@1",
            "hit_rate@5",
            "hit_rate@10",
            "hit_rate@20",
            "hit_rate@50",
            "hit_rate@100",
            "hit_rate@1000",
            #
            "precision",
            "precision@1",
            "precision@5",
            "precision@10",
            "precision@20",
            "precision@50",
            "precision@100",
            "precision@1000",
            #
            "recall",
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@20",
            "recall@50",
            "recall@100",
            "recall@1000",
            #
            "f1",
            "f1@1",
            "f1@5",
            "f1@10",
            "f1@20",
            "f1@50",
            "f1@100",
            "f1@1000",
            #
            "mrr",
            "mrr@1",
            "mrr@5",
            "mrr@10",
            "mrr@20",
            "mrr@50",
            "mrr@100",
            "mrr@1000",
            #
            "map",
            "map@1",
            "map@5",
            "map@10",
            "map@20",
            "map@50",
            "map@100",
            "map@1000",
            #
            "ndcg",
            "ndcg@1",
            "ndcg@5",
            "ndcg@10",
            "ndcg@20",
            "ndcg@50",
            "ndcg@100",
            "ndcg@1000",
            #
            "rbp.99",
            "rbp.95",
            "rbp.90",
            "rbp.80",
            "rbp.50",
            #
            "r-precision",
            #
            "bpref",
        ],
        return_mean=True,
        save_results_in_run=False,
    )

    with open(runcard_path, "r") as f:
        runcard = yaml.load(f, Loader=yaml.Loader)

    # Update runcard
    runcard["run"]["results"] = {
        k.upper()
        .replace("RECALL", "Recall")
        .replace("HIT_RATE", "HR")
        .replace("BPREF", "BPref")
        .replace("PRECISION", "P")
        .replace("R-PRECISION", "R-Prec"): float(v)
        for k, v in metrics.items()
    }

    content = {"metadata": runcard, "run": run.to_dict()}

    with open(runcard_path, "w") as f:
        runcard = f.write(yaml.dump(runcard, sort_keys=False))

    save_lz4(path, content)
