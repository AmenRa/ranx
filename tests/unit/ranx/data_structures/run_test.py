import pandas as pd
import pytest
from numba.typed import List
from pandas.testing import assert_frame_equal

from ranx import Qrels, Run
from ranx.data_structures.run import get_file_kind


# FIXTURES =====================================================================
@pytest.fixture
def run():
    return {
        "q1": {"d1": 0.1, "d2": 0.2, "d3": 0.3},
        "q2": {"d1": 0.1, "d2": 0.2},
    }


# TEST =========================================================================
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

    assert run.sorted is False

    run.sort()

    assert run.sorted is True

    assert List(run.run["q1"].keys()) == List(["d3", "d2", "d1"])
    assert List(run.run["q1"].values()) == List([3, 2, 1])
    assert List(run.run["q2"].keys()) == List(["d1", "d2"])
    assert List(run.run["q2"].values()) == List([2, 1])

    run.add_score("q2", "d2", 3)

    assert run.sorted is False


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


def test_to_dict():
    run_dict = {"q1": {"d1": 0.1, "d2": 0.2, "d3": 0.3}, "q2": {"d1": 0.1, "d2": 0.2}}

    assert Run(run_dict).to_dict() == run_dict


def test_to_dataframe():
    run_df = pd.DataFrame.from_dict(
        {
            "q_id": ["q1", "q1", "q1", "q2", "q2"],
            "doc_id": ["d1", "d2", "d3", "d1", "d2"],
            "score": [0.1, 0.2, 0.3, 0.1, 0.2],
        }
    )

    new_run_df = Run.from_df(run_df).to_dataframe()

    assert "q_id" in new_run_df.columns
    assert "doc_id" in new_run_df.columns
    assert "score" in new_run_df.columns

    assert_frame_equal(
        run_df.sort_values(by=run_df.columns.tolist()).reset_index(drop=True),
        new_run_df.sort_values(by=new_run_df.columns.tolist()).reset_index(drop=True),
    )


def test_save_load_json(run):
    Run(run).save("tests/unit/ranx/test_data/run.json")
    run = Run.from_file("tests/unit/ranx/test_data/run.json", name="test_run")

    assert run.name == "test_run"
    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 0.1
    assert run.run["q1"]["d2"] == 0.2
    assert run.run["q1"]["d3"] == 0.3
    assert run.run["q2"]["d1"] == 0.1
    assert run.run["q2"]["d2"] == 0.2


def test_save_load_trec(run):
    Run(run).save("tests/unit/ranx/test_data/run.trec")
    run = Run.from_file("tests/unit/ranx/test_data/run.trec", name="test_run")

    assert run.name == "test_run"
    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 0.1
    assert run.run["q1"]["d2"] == 0.2
    assert run.run["q1"]["d3"] == 0.3
    assert run.run["q2"]["d1"] == 0.1
    assert run.run["q2"]["d2"] == 0.2


def test_load_gzipped_trec(run):
    run = Run.from_file("tests/unit/ranx/test_data/run.trec.gz", name="test_run")

    assert run.name == "test_run"
    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 0.1
    assert run.run["q1"]["d2"] == 0.2
    assert run.run["q1"]["d3"] == 0.3
    assert run.run["q2"]["d1"] == 0.1
    assert run.run["q2"]["d2"] == 0.2


def test_save_load_lz4(run):
    Run(run).save("tests/unit/ranx/test_data/run.lz4")
    run = Run.from_file("tests/unit/ranx/test_data/run.lz4")

    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 0.1
    assert run.run["q1"]["d2"] == 0.2
    assert run.run["q1"]["d3"] == 0.3
    assert run.run["q2"]["d1"] == 0.1
    assert run.run["q2"]["d2"] == 0.2


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

    run = Run.from_dict(run_py, name="test_run")

    assert run.name == "test_run"
    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1
    assert run.run["q1"]["d2"] == 2
    assert run.run["q1"]["d3"] == 3
    assert run.run["q2"]["d1"] == 1
    assert run.run["q2"]["d2"] == 2


def test_from_dataframe():
    df = pd.DataFrame.from_dict(
        {
            "q_id": ["q1", "q1", "q1", "q2", "q2"],
            "doc_id": ["d1", "d2", "d3", "d1", "d2"],
            "score": [1.1, 2.1, 3.1, 1.1, 2.1],
        }
    )

    run = Run.from_df(df, name="test_run")

    assert run.name == "test_run"
    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 1.1
    assert run.run["q1"]["d2"] == 2.1
    assert run.run["q1"]["d3"] == 3.1
    assert run.run["q2"]["d1"] == 1.1
    assert run.run["q2"]["d2"] == 2.1


def test_from_parquet():
    run = Run.from_parquet("tests/unit/ranx/test_data/run.parquet", name="test_run")

    assert run.name == "test_run"
    assert len(run.run) == 2
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert run.run["q1"]["d1"] == 0.1
    assert run.run["q1"]["d2"] == 0.2
    assert run.run["q1"]["d3"] == 0.3
    assert run.run["q2"]["d1"] == 0.1
    assert run.run["q2"]["d2"] == 0.2


def test_make_comparable():
    qrels = Qrels(
        {
            "q1": {"d1": 1, "d2": 2, "d3": 3},
            "q2": {"d1": 1, "d2": 2},
            "q3": {"d1": 1, "d2": 2},
        }
    )

    run = Run(
        {
            "q1": {"d1": 0.3, "d2": 0.1, "d3": 0.2},
            "q2": {"d1": 0.1, "d2": 0.2},
            "q4": {"d1": 0.1, "d2": 0.2},
        }
    )

    run = run.make_comparable(qrels)

    assert len(run.run) == 3
    assert len(run.run["q1"]) == 3
    assert len(run.run["q2"]) == 2
    assert len(run.run["q3"]) == 0
    assert run.run["q1"]["d1"] == 0.3
    assert run.run["q1"]["d2"] == 0.1
    assert run.run["q1"]["d3"] == 0.2
    assert run.run["q2"]["d1"] == 0.1
    assert run.run["q2"]["d2"] == 0.2


def test_get_file_kind():
    assert get_file_kind("qrels.json") == "json"
    assert get_file_kind("qrels.trec") == "trec"
    assert get_file_kind("qrels.txt") == "trec"
    assert get_file_kind("qrels.gz") == "gz"
    assert get_file_kind("qrels.lz4") == "lz4"
    assert get_file_kind("qrels.parquet") == "parquet"
    assert get_file_kind("qrels.parq") == "parquet"
