import pandas as pd
import pytest
from numba.typed import List

from ranx import Run


def test_init():
    run_dict = {
        "q1": {
            "d1": 1,
            "d2": 2,
            "d3": 3,
        },
        "q2": {
            "d1": 1,
            "d2": 2,
        },
    }

    run = Run(run_dict, name="bm25")

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1
    assert run.run["q1"]["d2"] == 2
    assert run.run["q1"]["d3"] == 3
    assert run.run["q2"]["d1"] == 1
    assert run.run["q2"]["d2"] == 2
    assert run.size == 2
    assert run.name == "bm25"


def test_size():
    run = Run()

    run.add_score("q1", "d1", 1)
    run.add_score("q1", "d2", 2)
    run.add_score("q1", "d3", 3)
    run.add_score("q2", "d1", 1)
    run.add_score("q2", "d2", 2)

    assert run.size == 2


def test_add_score():
    run = Run()

    run.add_score("q1", "d1", 1)
    run.add_score("q1", "d2", 2)
    run.add_score("q1", "d3", 3)
    run.add_score("q2", "d1", 1)
    run.add_score("q2", "d2", 2)

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1
    assert run.run["q1"]["d2"] == 2
    assert run.run["q1"]["d3"] == 3
    assert run.run["q2"]["d1"] == 1
    assert run.run["q2"]["d2"] == 2


def test_add_multi():
    run = Run()

    q_ids = ["q1", "q2"]
    doc_ids = [
        [
            "d1",
            "d2",
            "d3",
        ],
        [
            "d1",
            "d2",
        ],
    ]

    scores = [[1, 2, 3], [1, 2]]

    run.add_multi(q_ids, doc_ids, scores)

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1
    assert run.run["q1"]["d2"] == 2
    assert run.run["q1"]["d3"] == 3
    assert run.run["q2"]["d1"] == 1
    assert run.run["q2"]["d2"] == 2

    run = Run()

    q_ids = ["q1", "q2"]
    doc_ids = [
        [
            "d1",
            "d2",
            "d3",
        ],
        [
            "d1",
            "d2",
        ],
    ]

    scores = [[1, 2, 3], [1, 2]]

    run.add_multi(q_ids, doc_ids, scores)

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1
    assert run.run["q1"]["d2"] == 2
    assert run.run["q1"]["d3"] == 3
    assert run.run["q2"]["d1"] == 1
    assert run.run["q2"]["d2"] == 2


def test_sort():
    run = Run()

    run.add_score("q1", "d1", 1)
    run.add_score("q1", "d2", 2)
    run.add_score("q1", "d3", 3)
    run.add_score("q2", "d1", 2)
    run.add_score("q2", "d2", 1)

    assert run.sorted == False

    run.sort()

    assert run.sorted == True

    assert List(run.run["q1"].keys()) == List(["d3", "d2", "d1"])
    assert List(run.run["q1"].values()) == List([3, 2, 1])
    assert List(run.run["q2"].keys()) == List(["d1", "d2"])
    assert List(run.run["q2"].values()) == List([2, 1])

    run.add_score("q2", "d2", 3)

    assert run.sorted == False


def test_to_typed_list():
    run = Run()

    run.add_score("q1", "d1", 1)
    run.add_score("q1", "d2", 2)
    run.add_score("q1", "d3", 3)
    run.add_score("q2", "d1", 2)
    run.add_score("q2", "d2", 1)

    run_list = run.to_typed_list()

    assert len(run_list) == 2
    assert run_list[0].shape == (3, 2)
    assert run_list[1].shape == (2, 2)


def test_from_dict():
    run_py = {
        "q1": {
            "d1": 1,
            "d2": 2,
            "d3": 3,
        },
        "q2": {
            "d1": 1,
            "d2": 2,
        },
    }

    run = Run.from_dict(run_py)

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1
    assert run.run["q1"]["d2"] == 2
    assert run.run["q1"]["d3"] == 3
    assert run.run["q2"]["d1"] == 1
    assert run.run["q2"]["d2"] == 2


def test_from_trec_file():
    run = Run.from_file("tests/unit/ranx/test_data/run.trec", kind="trec")

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 0.1
    assert run.run["q1"]["d2"] == 0.2
    assert run.run["q1"]["d3"] == 0.3
    assert run.run["q2"]["d1"] == 0.1
    assert run.run["q2"]["d2"] == 0.2


def test_from_json_file():
    run = Run.from_file("tests/unit/ranx/test_data/run.json")

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 0.1
    assert run.run["q1"]["d2"] == 0.2
    assert run.run["q1"]["d3"] == 0.3
    assert run.run["q2"]["d1"] == 0.1
    assert run.run["q2"]["d2"] == 0.2


def test_from_dataframe():
    df = pd.DataFrame.from_dict(
        {
            "q_id": ["q1", "q1", "q1", "q2", "q2"],
            "doc_id": ["d1", "d2", "d3", "d1", "d2"],
            "score": [1.1, 2.1, 3.1, 1.1, 2.1],
        }
    )

    run = Run.from_df(df)

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1.1
    assert run.run["q1"]["d2"] == 2.1
    assert run.run["q1"]["d3"] == 3.1
    assert run.run["q2"]["d1"] == 1.1
    assert run.run["q2"]["d2"] == 2.1
