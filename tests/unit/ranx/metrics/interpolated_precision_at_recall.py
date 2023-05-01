import random
from math import isclose

import numpy as np
import pytest
import pytrec_eval

from ranx import Qrels, Run
from ranx.metrics.interpolated_precision_at_recall import (
    interpolated_precision_at_recall,
)


# Wrapper for pytrec_eval
def run_trec_iprec_at_recall(qrels, run):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"iprec_at_recall"})
    results = evaluator.evaluate(run)

    matrix_results = np.zeros((len(results), 11))

    for i, q_id in enumerate(results):
        matrix_results[i, 0] = results[q_id]["iprec_at_recall_0.00"]
        matrix_results[i, 1] = results[q_id]["iprec_at_recall_0.10"]
        matrix_results[i, 2] = results[q_id]["iprec_at_recall_0.20"]
        matrix_results[i, 3] = results[q_id]["iprec_at_recall_0.30"]
        matrix_results[i, 4] = results[q_id]["iprec_at_recall_0.40"]
        matrix_results[i, 5] = results[q_id]["iprec_at_recall_0.50"]
        matrix_results[i, 6] = results[q_id]["iprec_at_recall_0.60"]
        matrix_results[i, 7] = results[q_id]["iprec_at_recall_0.70"]
        matrix_results[i, 8] = results[q_id]["iprec_at_recall_0.80"]
        matrix_results[i, 9] = results[q_id]["iprec_at_recall_0.90"]
        matrix_results[i, 10] = results[q_id]["iprec_at_recall_1.00"]

    return matrix_results


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

ranx_qrels = Qrels.from_dict(trec_qrels).to_typed_list()
ranx_run = Run.from_dict(trec_run).to_typed_list()


def test_iprec():
    trec_scores = run_trec_iprec_at_recall(trec_qrels, trec_run)
    ranx_scores = interpolated_precision_at_recall(ranx_qrels, ranx_run)

    assert np.allclose(trec_scores.mean(axis=0), ranx_scores.mean(axis=0))
