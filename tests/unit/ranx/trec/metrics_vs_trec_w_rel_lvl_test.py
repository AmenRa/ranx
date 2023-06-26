import random
from math import isclose

import numpy as np
import pytest
import pytrec_eval

from ranx import Qrels, Run, evaluate


# Wrapper for pytrec_eval
def run_trec_metrics(qrels, run, metrics, rel_lvl):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics, rel_lvl)
    results = evaluator.evaluate(run)
    return {m: np.mean([v[m] for v in results.values()]) for m in list(metrics)}


def run_single_trec_metric(qrels, run, metric, rel_lvl):
    return run_trec_metrics(qrels, run, {metric}, rel_lvl)[metric]


def generate_qrels(query_count, max_relevant_per_query):
    qrels = {}
    for i in range(query_count):
        k = random.choice(range(1, max_relevant_per_query))
        y_t = {f"d{j}": random.choice([0, 1, 2, 3, 4, 5]) for j in range(k)}
        qrels[f"q{i}"] = y_t

    return qrels


def generate_run(query_count, max_result_count):
    run = {}
    for i in range(query_count):
        result_count = random.choice(range(1, max_result_count))
        y_p = {f"d{j}": random.uniform(0.0, 1.0) for j in range(result_count)}
        run[f"q{i}"] = y_p

    return run


random.seed = 42
np.random.seed(42)

query_count = 10_000
max_relevant_per_query = 10
max_result_count = 100

trec_qrels = generate_qrels(query_count, max_relevant_per_query)
trec_run = generate_run(query_count, max_result_count)

re_qrels = Qrels.from_dict(trec_qrels).to_typed_list()
re_run = Run.from_dict(trec_run).to_typed_list()

rel_lvl = 2


def test_precision():
    trec_score = run_single_trec_metric(trec_qrels, trec_run, "P_5", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"precision@5-l{rel_lvl}")

    assert isclose(re_score, trec_score)

    trec_score = run_single_trec_metric(trec_qrels, trec_run, "P_10", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"precision@10-l{rel_lvl}")

    assert isclose(re_score, trec_score)

    trec_score = run_single_trec_metric(trec_qrels, trec_run, "P_100", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"precision@100-l{rel_lvl}")

    assert isclose(re_score, trec_score)


def test_recall():
    trec_score = run_single_trec_metric(trec_qrels, trec_run, "recall_5", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"recall@5-l{rel_lvl}")

    assert isclose(re_score, trec_score)

    trec_score = run_single_trec_metric(trec_qrels, trec_run, "recall_10", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"recall@10-l{rel_lvl}")

    assert isclose(re_score, trec_score)

    trec_score = run_single_trec_metric(trec_qrels, trec_run, "recall_100", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"recall@100-l{rel_lvl}")

    assert isclose(re_score, trec_score)


def test_r_precision():
    trec_score = run_single_trec_metric(trec_qrels, trec_run, "Rprec", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"r-precision-l{rel_lvl}")

    assert isclose(re_score, trec_score)


def test_mrr():
    trec_score = run_single_trec_metric(trec_qrels, trec_run, "recip_rank", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"mrr@100-l{rel_lvl}")

    assert isclose(re_score, trec_score)


def test_map():
    trec_score = run_single_trec_metric(trec_qrels, trec_run, "map_cut_5", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"map@5-l{rel_lvl}")

    assert isclose(re_score, trec_score)

    trec_score = run_single_trec_metric(trec_qrels, trec_run, "map_cut_10", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"map@10-l{rel_lvl}")

    assert isclose(re_score, trec_score)

    trec_score = run_single_trec_metric(trec_qrels, trec_run, "map_cut_100", rel_lvl)
    re_score = evaluate(re_qrels, re_run, f"map@100-l{rel_lvl}")

    assert isclose(re_score, trec_score)


# TREC EVAL ISSUE? -------------------------------------------------------------
# def test_ndcg():
#     trec_score = run_single_trec_metric(trec_qrels, trec_run, "ndcg_cut_5", rel_lvl)
#     re_score = evaluate(re_qrels, re_run, f"ndcg@5-l{rel_lvl}")

#     assert isclose(re_score, trec_score)

#     trec_score = run_single_trec_metric(trec_qrels, trec_run, "ndcg_cut_10", rel_lvl)
#     re_score = evaluate(re_qrels, re_run, f"ndcg@10-l{rel_lvl}")

#     assert isclose(re_score, trec_score)

#     trec_score = run_single_trec_metric(trec_qrels, trec_run, "ndcg_cut_100", rel_lvl)
#     re_score = evaluate(re_qrels, re_run, f"ndcg@100-l{rel_lvl}")

#     assert isclose(re_score, trec_score)
