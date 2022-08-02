import random

import pytest
from ranx import Qrels, Run, compare


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

# FIXTURES =====================================================================
@pytest.fixture
def query_count():
    return 1_000


@pytest.fixture
def max_relevant_per_query():
    return 10


@pytest.fixture
def max_result_count():
    return 100


@pytest.fixture
def qrels(query_count, max_relevant_per_query):
    qrels = generate_qrels(query_count, max_relevant_per_query)
    return Qrels.from_dict(qrels)


@pytest.fixture
def runs(query_count, max_result_count):
    runs = []

    for i in range(1, 4):
        run = generate_run(query_count, max_result_count)
        run = Run.from_dict(run)
        run.name = f"model_{i}"
        runs.append(run)

    return runs


@pytest.fixture
def metrics():
    return ["map@100", "mrr@100", "ndcg@10"]


# TESTS ========================================================================
def test_report(qrels, runs, metrics):
    report = compare(qrels, runs, metrics)

    assert report.model_names == [run.name for run in runs]
    assert report.metrics == metrics


def test_to_dict(qrels, runs, metrics):
    report = compare(qrels, runs, metrics)
    report_dict = report.to_dict()

    assert "stat_test" in report_dict
    assert report_dict["stat_test"] == report.stat_test
    assert "model_names" in report_dict
    assert report_dict["model_names"] == report.model_names
    assert "metrics" in report_dict
    assert report_dict["metrics"] == report.metrics
    assert all(run.name in report_dict for run in runs)
    assert all(
        "scores" in report_dict[model]
        and "comparisons" in report_dict[model]
        and "win_tie_loss"
        for model in report_dict["model_names"]
    )
    assert all(
        all(
            metric in report_dict[model]["scores"]
            for metric in report_dict["metrics"]
        )
        for model in report_dict["model_names"]
    )
    assert all(
        all(
            all(
                metric in report_dict[m1]["comparisons"][m2]
                for metric in report_dict["metrics"]
            )
            for m2 in report_dict["model_names"]
            if m1 != m2
        )
        for m1 in report_dict["model_names"]
    )
    assert all(
        all(
            all(
                all(
                    x in report_dict[m1]["win_tie_loss"][m2][metric]
                    for x in ["W", "T", "L"]
                )
                for metric in report_dict["metrics"]
            )
            for m2 in report_dict["model_names"]
            if m1 != m2
        )
        for m1 in report_dict["model_names"]
    )


def test_stat_test(qrels, runs, metrics):
    report = compare(qrels, runs, metrics)
    assert report.stat_test == "student"
    assert (
        report.get_stat_test_label(report.stat_test)
        == "paired Student's t-test"
    )
    assert report.get_stat_test_label(report.stat_test) in report.to_latex()

    report = compare(qrels, runs, metrics, stat_test="student")
    assert report.stat_test == "student"
    assert (
        report.get_stat_test_label(report.stat_test)
        == "paired Student's t-test"
    )
    assert report.get_stat_test_label(report.stat_test) in report.to_latex()

    report = compare(qrels, runs, metrics, stat_test="fisher")
    assert report.stat_test == "fisher"
    assert (
        report.get_stat_test_label(report.stat_test)
        == "Fisher's randomization test"
    )
    assert report.get_stat_test_label(report.stat_test) in report.to_latex()

    report = compare(qrels, runs, metrics, stat_test="tukey")
    assert report.stat_test == "tukey"
    assert report.get_stat_test_label(report.stat_test) == "Tukey's HSD test"
    assert report.get_stat_test_label(report.stat_test) in report.to_latex()
