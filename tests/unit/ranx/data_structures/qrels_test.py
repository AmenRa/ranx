import pandas as pd
import pytest
from numba.typed import List

from ranx import Qrels


def test_init():
    qrels_dict = {
        "q1": {"d1": 1, "d2": 2, "d3": 3},
        "q2": {"d1": 1, "d2": 2},
    }

    qrels = Qrels(qrels_dict, name="rocco")

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2
    assert qrels.size == 2
    assert qrels.name == "rocco"


def test_size():
    qrels = Qrels()

    qrels.add_score("q1", "d1", 1)
    qrels.add_score("q1", "d2", 2)
    qrels.add_score("q1", "d3", 3)
    qrels.add_score("q2", "d1", 1)
    qrels.add_score("q2", "d2", 2)

    assert qrels.size == 2


def test_add_score():
    qrels = Qrels()

    qrels.add_score("q1", "d1", 1)
    qrels.add_score("q1", "d2", 2)
    qrels.add_score("q1", "d3", 3)
    qrels.add_score("q2", "d1", 1)
    qrels.add_score("q2", "d2", 2)

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2


def test_add_multi():
    qrels = Qrels()

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

    qrels.add_multi(q_ids, doc_ids, scores)

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2

    qrels = Qrels()

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

    qrels.add_multi(q_ids, doc_ids, scores)

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2


def test_sort():
    qrels = Qrels()

    qrels.add_score("q1", "d1", 1)
    qrels.add_score("q1", "d2", 2)
    qrels.add_score("q1", "d3", 3)
    qrels.add_score("q2", "d1", 2)
    qrels.add_score("q2", "d2", 1)

    assert qrels.sorted == False

    qrels.sort()

    assert qrels.sorted == True

    assert List(qrels.qrels["q1"].keys()) == List(["d3", "d2", "d1"])
    assert List(qrels.qrels["q1"].values()) == List([3, 2, 1])
    assert List(qrels.qrels["q2"].keys()) == List(["d1", "d2"])
    assert List(qrels.qrels["q2"].values()) == List([2, 1])

    qrels.add_score("q2", "d2", 3)

    assert qrels.sorted == False


def test_to_typed_list():
    qrels = Qrels()

    qrels.add_score("q1", "d1", 1)
    qrels.add_score("q1", "d2", 2)
    qrels.add_score("q1", "d3", 3)
    qrels.add_score("q2", "d1", 2)
    qrels.add_score("q2", "d2", 1)

    qrels_list = qrels.to_typed_list()

    assert len(qrels_list) == 2
    assert qrels_list[0].shape == (3, 2)
    assert qrels_list[1].shape == (2, 2)


def test_from_dict():
    qrels_py = {
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

    qrels = Qrels.from_dict(qrels_py)

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2


def test_from_trec_file():
    qrels = Qrels.from_file("tests/unit/ranx/test_data/qrels.trec", kind="trec")

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2


def test_from_json_file():
    qrels = Qrels.from_file("tests/unit/ranx/test_data/qrels.json")

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2


def test_from_dataframe():
    df = pd.DataFrame.from_dict(
        {
            "q_id": ["q1", "q1", "q1", "q2", "q2"],
            "doc_id": ["d1", "d2", "d3", "d1", "d2"],
            "score": [1, 2, 3, 1, 2],
        }
    )

    qrels = Qrels.from_df(df)

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2


def test_set_relevance_level():
    qrels_py = {
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

    qrels = Qrels.from_dict(qrels_py)
    qrels.set_relevance_level(2)

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 0
    assert qrels.qrels["q1"]["d2"] == 1
    assert qrels.qrels["q1"]["d3"] == 2
    assert qrels.qrels["q2"]["d1"] == 0
    assert qrels.qrels["q2"]["d2"] == 1
