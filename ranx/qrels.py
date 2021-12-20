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


class Qrels(object):
    def __init__(self):
        self.qrels = TypedDict.empty(
            key_type=types.unicode_type,
            value_type=types.DictType(types.unicode_type, types.int64),
        )
        self.sorted = False
        self.name = None

    def keys(self):
        """Returns query ids. Used internally."""
        return self.qrels.keys()

    def add_score(self, q_id, doc_id, score):
        """Add a (doc_id, score) pair to a query (or, change its value if it already exists)."""
        if self.qrels.get(q_id) is None:
            self.qrels[q_id] = TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.int64,
            )
        self.qrels[q_id][doc_id] = int(score)
        self.sorted = False

    def add(self, q_id, doc_ids, scores):
        """Add a query."""
        self.add_multi([q_id], [doc_ids], [scores])

    def add_multi(
        self,
        q_ids: List[str],
        doc_ids: List[List[str]],
        scores: List[List[int]],
    ):
        """Add multiple queries at once."""
        q_ids = TypedList(q_ids)
        doc_ids = TypedList([TypedList(x) for x in doc_ids])
        scores = TypedList([TypedList(map(int, x)) for x in scores])

        self.qrels = add_and_sort(self.qrels, q_ids, doc_ids, scores)
        self.sorted = True

    def get_query_ids(self):
        """Returns query ids."""
        return list(self.qrels.keys())

    def get_doc_ids_and_scores(self):
        """Returns doc ids and relevance judgments."""
        return list(self.qrels.values())

    # Sort in place
    def sort(self):
        """Sort. Used internally."""
        self.qrels = sort_dict_by_key(self.qrels)
        self.qrels = sort_dict_of_dict_by_value(self.qrels)
        self.sorted = True

    def to_typed_list(self):
        """Convert Qrels to Numba Typed List. Used internally."""
        if self.sorted == False:
            self.sort()
        return to_typed_list(self.qrels)

    def save(self, path: str = "qrels.txt"):
        """Write `qrels` to `path` in TREC qrels format."""
        with open(path, "w") as f:
            for i, q_id in enumerate(self.qrels.keys()):
                for j, doc_id in enumerate(self.qrels[q_id].keys()):
                    score = self.qrels[q_id][doc_id]
                    f.write(f"{q_id} 0 {doc_id} {score}")

                    if (
                        i != len(self.qrels.keys()) - 1
                        or j != len(self.qrels[q_id].keys()) - 1
                    ):
                        f.write("\n")

    @property
    def size(self):
        return len(self.qrels)

    @staticmethod
    def from_dict(d: Dict[str, Dict[str, int]]):
        """Convert a Python dictionary in form of {q_id: {doc_id: rel_score}} to a ranx.Qrels."""
        q_ids = list(d.keys())
        doc_ids = [list(doc.keys()) for doc in d.values()]
        scores = [list(doc.values()) for doc in d.values()]

        qrels = Qrels()

        qrels.add_multi(q_ids, doc_ids, scores)

        return qrels

    @staticmethod
    def from_file(path: str, type: str = "trec"):
        """Parse a qrels file into ranx.Qrels."""
        assert type in {
            "trec",
            "json",
        }, "Error `type` must be 'trec' of 'json'"

        if type == "trec":
            qrels = defaultdict(dict)
            with open(path) as f:
                for line in f:
                    q_id, _, doc_id, rel = line.split()
                    qrels[q_id][doc_id] = int(rel)
        else:
            qrels = json.loads(open(path, "r").read())

        return Qrels.from_dict(qrels)

    @staticmethod
    def from_df(
        df: pd.DataFrame,
        q_id_col: str = "q_id",
        doc_id_col: str = "doc_id",
        score_col: str = "score",
    ):
        """Convert a Pandas DataFrame to ranx.Qrels."""
        assert (
            df[q_id_col].dtype == "O"
        ), "DataFrame scores column dtype must be `object` (string)"
        assert (
            df[doc_id_col].dtype == "O"
        ), "DataFrame scores column dtype must be `object` (string)"
        assert df[score_col].dtype == int, "DataFrame scores column dtype must be `int`"

        qrels_dict = (
            df.groupby(q_id_col)[[doc_id_col, score_col]]
            .apply(lambda g: {x[0]: x[1] for x in g.values.tolist()})
            .to_dict()
        )

        return Qrels.from_dict(qrels_dict)

    def __getitem__(self, q_id):
        return dict(self.qrels[q_id])

    # def __setitem__(self, q_id, x):
    #     self.qrels[q_id] = x

    def __len__(self) -> int:
        return len(self.qrels)

    def __repr__(self):
        return self.qrels.__repr__()

    def __str__(self):
        return self.qrels.__str__()
