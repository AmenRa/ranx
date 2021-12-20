import json
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from numba import types
from numba.typed import Dict as TypedDict
from numba.typed import List as TypedList

from .qrels_run_common import (
    add_and_sort,
    sort_dict_by_key,
    sort_dict_of_dict_by_value,
    to_typed_list,
)


class Run(object):
    def __init__(self):
        self.run = TypedDict.empty(
            key_type=types.unicode_type,
            value_type=types.DictType(types.unicode_type, types.float64),
        )
        self.sorted = False
        self.name = None
        self.scores = defaultdict(dict)
        self.mean_scores = {}

    def keys(self):
        """Returns query ids. Used internally."""
        return self.run.keys()

    def add_score(self, q_id, doc_id, score):
        """Add a (doc_id, score) pair to a query."""
        if self.run.get(q_id) is None:
            self.run[q_id] = TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.float64,
            )
        self.run[q_id][doc_id] = float(score)
        self.sorted = False

    def add(self, q_id: str, doc_ids: List[str], scores: List[float]):
        """Add a query."""
        self.add_multi([q_id], [doc_ids], [scores])

    def add_multi(
        self,
        q_ids: List[str],
        doc_ids: List[List[str]],
        scores: List[List[float]],
    ):
        """Add multiple queries at once."""
        q_ids = TypedList(q_ids)
        doc_ids = TypedList([TypedList(x) for x in doc_ids])
        scores = TypedList([TypedList(map(float, x)) for x in scores])

        self.run = add_and_sort(self.run, q_ids, doc_ids, scores)
        self.sorted = True

    def get_query_ids(self):
        """Returns query ids."""
        return list(self.run.keys())

    def get_doc_ids_and_scores(self):
        """Returns doc ids and relevance scores."""
        return list(self.run.values())

    # Sort in place
    def sort(self):
        """Sort. Used internally."""
        self.run = sort_dict_by_key(self.run)
        self.run = sort_dict_of_dict_by_value(self.run)
        self.sorted = True

    def to_typed_list(self):
        """Convert Run to Numba Typed List. Used internally."""
        if self.sorted == False:
            self.sort()
        return to_typed_list(self.run)

    def save(self, path: str = "run.txt"):
        """Write `run` to `path` in TREC run format."""
        if self.sorted == False:
            self.sort()
        with open(path, "w") as f:
            for i, q_id in enumerate(self.run.keys()):
                for rank, doc_id in enumerate(self.run[q_id].keys()):
                    score = self.run[q_id][doc_id]
                    f.write(f"{q_id} Q0 {doc_id} {rank+1} {score} {self.name}")

                    if (
                        i != len(self.run.keys()) - 1
                        or rank != len(self.run[q_id].keys()) - 1
                    ):
                        f.write("\n")

    @staticmethod
    def from_dict(d: Dict[str, Dict[str, float]]):
        """Convert a Python dictionary in form of {q_id: {doc_id: rank_score}} to a ranx.Run."""
        q_ids = list(d.keys())
        doc_ids = [list(doc.keys()) for doc in d.values()]
        scores = [list(doc.values()) for doc in d.values()]

        run = Run()

        run.add_multi(q_ids, doc_ids, scores)

        return run

    @staticmethod
    def from_file(path: str, type: str = "trec"):
        """Parse a run file into ranx.Run."""
        assert type in {
            "trec",
            "json",
        }, "Error `type` must be 'trec' of 'json'"

        if type == "trec":
            run = defaultdict(dict)
            name = ""
            with open(path) as f:
                for line in f:
                    q_id, _, doc_id, _, rel, run_name = line.split()
                    run[q_id][doc_id] = float(rel)
                    if name == "":
                        name = run_name
        else:
            run = json.loads(open(path, "r").read())

        run = Run.from_dict(run)

        if type == "trec":
            run.name = name

        return run

    @staticmethod
    def from_df(
        df: pd.DataFrame,
        q_id_col: str = "q_id",
        doc_id_col: str = "doc_id",
        score_col: str = "score",
    ):
        """Convert a Pandas DataFrame to ranx.Run."""
        assert (
            df[q_id_col].dtype == "O"
        ), "DataFrame scores column dtype must be `object` (string)"
        assert (
            df[doc_id_col].dtype == "O"
        ), "DataFrame scores column dtype must be `object` (string)"
        assert (
            df[score_col].dtype == float
        ), "DataFrame scores column dtype must be `float`"

        run_py = (
            df.groupby(q_id_col)[[doc_id_col, score_col]]
            .apply(lambda g: {x[0]: x[1] for x in g.values.tolist()})
            .to_dict()
        )

        return Run.from_dict(run_py)

    @property
    def size(self):
        return len(self.run)

    def __getitem__(self, q_id):
        return dict(self.run[q_id])

    def __len__(self) -> int:
        return len(self.run)

    def __repr__(self):
        return self.run.__repr__()

    def __str__(self):
        return self.run.__str__()
