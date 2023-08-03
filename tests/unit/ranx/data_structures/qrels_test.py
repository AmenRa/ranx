import pandas as pd
from numba.typed import List
from pandas.testing import assert_frame_equal

from ranx import Qrels
from ranx.data_structures.qrels import get_file_kind


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

    assert qrels.sorted is False

    qrels.sort()

    assert qrels.sorted is True

    assert List(qrels.qrels["q1"].keys()) == List(["d3", "d2", "d1"])
    assert List(qrels.qrels["q1"].values()) == List([3, 2, 1])
    assert List(qrels.qrels["q2"].keys()) == List(["d1", "d2"])
    assert List(qrels.qrels["q2"].values()) == List([2, 1])

    qrels.add_score("q2", "d2", 3)

    assert qrels.sorted is False


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


def test_to_dict():
    qrels_dict = {"q1": {"d1": 1, "d2": 2, "d3": 3}, "q2": {"d1": 1, "d2": 2}}

    assert Qrels(qrels_dict).to_dict() == qrels_dict


def test_to_dataframe():
    qrels_df = pd.DataFrame.from_dict(
        {
            "q_id": ["q1", "q1", "q1", "q2", "q2"],
            "doc_id": ["d1", "d2", "d3", "d1", "d2"],
            "score": [1, 2, 3, 1, 2],
        }
    )

    new_qrels_df = Qrels.from_df(qrels_df).to_dataframe()

    assert "q_id" in new_qrels_df.columns
    assert "doc_id" in new_qrels_df.columns
    assert "score" in new_qrels_df.columns

    assert_frame_equal(
        qrels_df.sort_values(by=qrels_df.columns.tolist()).reset_index(drop=True),
        new_qrels_df.sort_values(by=new_qrels_df.columns.tolist()).reset_index(
            drop=True
        ),
    )


def test_from_dict():
    qrels_py = {"q1": {"d1": 1, "d2": 2, "d3": 3}, "q2": {"d1": 1, "d2": 2}}

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
    qrels = Qrels.from_file("tests/unit/ranx/test_data/qrels.trec")

    assert len(qrels.qrels) == 2
    assert len(qrels.qrels["q1"]) == 3
    assert len(qrels.qrels["q2"]) == 2
    assert qrels.qrels["q1"]["d1"] == 1
    assert qrels.qrels["q1"]["d2"] == 2
    assert qrels.qrels["q1"]["d3"] == 3
    assert qrels.qrels["q2"]["d1"] == 1
    assert qrels.qrels["q2"]["d2"] == 2


def test_from_gzipped_trec_file():
    qrels = Qrels.from_file("tests/unit/ranx/test_data/qrels.trec.gz")

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


def test_from_parquet():
    qrels = Qrels.from_parquet("tests/unit/ranx/test_data/qrels.parquet")

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


def test_get_file_kind():
    assert get_file_kind("qrels.json") == "json"
    assert get_file_kind("qrels.trec") == "trec"
    assert get_file_kind("qrels.txt") == "trec"
    assert get_file_kind("qrels.gz") == "gz"
    assert get_file_kind("qrels.parquet") == "parquet"
    assert get_file_kind("qrels.parq") == "parquet"
