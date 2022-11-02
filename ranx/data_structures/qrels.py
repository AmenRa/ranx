import json
import os
from collections import defaultdict
from typing import Dict, List

import ir_datasets
import numpy as np
import orjson
import pandas as pd
from numba import njit, prange, types
from numba.typed import Dict as TypedDict
from numba.typed import List as TypedList

from .common import (
    add_and_sort,
    create_and_sort,
    sort_dict_by_key,
    sort_dict_of_dict_by_value,
    to_typed_list,
)


@njit(cache=True, parallel=True)
def _set_relevance_level(qrels, rel_lvl):
    q_ids = TypedList(qrels.keys())
    results = TypedList(qrels.values())

    for i in prange(len(q_ids)):
        doc_ids = TypedList(results[i].keys())
        scores = TypedList(results[i].values())

        for j in range(len(doc_ids)):
            qrels[q_ids[i]][doc_ids[j]] = scores[j] - (rel_lvl - 1)

    return qrels


class Qrels(object):
    """`Qrels`, or _query relevance judgments_, stores the ground truth for conducting evaluations.<\br>
    The preferred way for creating a `Qrels` istance is converting Python dictionary as follows:

    ```python
    qrels_dict = {
        "q_1": {
            "d_1": 1,
            "d_2": 2,
        },
        "q_2": {
            "d_3": 2,
            "d_2": 1,
            "d_5": 3,
        },
    }

    qrels = Qrels(qrels_dict, name="MSMARCO")

    qrels = Qrels()  # Creates an empty Qrels with no name
    ```
    """

    def __init__(
        self, qrels: Dict[str, Dict[str, int]] = None, name: str = None
    ):
        if qrels is None:
            self.qrels = TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.DictType(types.unicode_type, types.int64),
            )
            self.sorted = False
        else:
            # Query IDs
            q_ids = list(qrels.keys())
            q_ids = TypedList(q_ids)

            # Doc IDs
            doc_ids = [list(doc.keys()) for doc in qrels.values()]
            max_len = max(len(y) for x in doc_ids for y in x)
            dtype = f"<U{max_len}"
            doc_ids = TypedList([np.array(x, dtype=dtype) for x in doc_ids])

            # Scores
            scores = [list(doc.values()) for doc in qrels.values()]
            scores = TypedList([np.array(x, dtype=int) for x in scores])

            self.qrels = create_and_sort(q_ids, doc_ids, scores)
            self.sorted = True

        self.name = name

    def keys(self):
        """Returns query ids. Used internally."""
        return self.qrels.keys()

    def add_score(self, q_id: str, doc_id: str, score: int):
        """Add a (doc_id, score) pair to a query (or, change its value if it already exists).

        Args:
            q_id (str): Query ID
            doc_id (str): Document ID
            score (int): Relevance score judgment
        """
        if self.qrels.get(q_id) is None:
            self.qrels[q_id] = TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.int64,
            )
        self.qrels[q_id][doc_id] = int(score)
        self.sorted = False

    def add(self, q_id: str, doc_ids: List[str], scores: List[int]):
        """Add a query and its relevant documents with the associated relevance score judgment.

        Args:
            q_id (str): Query ID
            doc_ids (List[str]): List of Document IDs
            scores (List[int]): List of relevance score judgments
        """
        self.add_multi([q_id], [doc_ids], [scores])

    def add_multi(
        self,
        q_ids: List[str],
        doc_ids: List[List[str]],
        scores: List[List[int]],
    ):
        """Add multiple queries at once.

        Args:
            q_ids (List[str]): List of Query IDs
            doc_ids (List[List[str]]): List of list of Document IDs
            scores (List[List[int]]): List of list of relevance score judgments
        """
        q_ids = TypedList(q_ids)
        doc_ids = TypedList([TypedList(x) for x in doc_ids])
        scores = TypedList([TypedList(map(int, x)) for x in scores])

        self.qrels = add_and_sort(self.qrels, q_ids, doc_ids, scores)
        self.sorted = True

    def set_relevance_level(self, rel_lvl: int = 1):
        """Sets relevance level."""
        self.qrels = _set_relevance_level(self.qrels, rel_lvl)

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

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Convert Qrels to Python dictionary.

        Returns:
            Dict[str, Dict[str, int]]: Qrels as Python dictionary
        """
        d = defaultdict(dict)
        for q_id in self.keys():
            d[q_id] = dict(self[q_id])
        return d

    def save(self, path: str = "qrels.json", kind: str = None):
        """Write `qrels` to `path` as JSON file or TREC qrels format.        File type is automatically inferred form the filename extension: ".json" -> "json", ".trec" -> "trec", ".txt" -> "trec". Use the "kind" argument to override this behavior.

        Args:
            path (str, optional): Saving path. Defaults to "qrels.json".
            kind (str, optional): Kind of file to save, must be either "json" or "trec". If None, it will be automatically inferred from the filename extension.
        """
        # Infer file extension -------------------------------------------------
        kind = get_file_kind(path, kind)

        # Save Qrels -----------------------------------------------------------
        if kind == "json":
            with open(path, "wb") as f:
                f.write(
                    orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2)
                )
        else:
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

    @staticmethod
    def from_dict(d: Dict[str, Dict[str, int]]):
        """Convert a Python dictionary in form of {q_id: {doc_id: score}} to ranx.Qrels.

        Args:
            d (Dict[str, Dict[str, int]]): Qrels as Python dictionary

        Returns:
            Qrels: ranx.Qrels
        """
        # Query IDs
        q_ids = list(d.keys())
        q_ids = TypedList(q_ids)

        # Doc IDs
        doc_ids = [list(doc.keys()) for doc in d.values()]
        max_len = max(len(y) for x in doc_ids for y in x)
        dtype = f"<U{max_len}"
        doc_ids = TypedList([np.array(x, dtype=dtype) for x in doc_ids])

        # Scores
        scores = [list(doc.values()) for doc in d.values()]
        scores = TypedList([np.array(x, dtype=int) for x in scores])

        qrels = Qrels()
        qrels.qrels = create_and_sort(q_ids, doc_ids, scores)
        qrels.sorted = True

        return qrels

    @staticmethod
    def from_file(path: str, kind: str = None):
        """Parse a qrels file into ranx.Qrels. Supported formats are JSON and TREC qrels format. Correct import behavior is inferred from the file extension: ".json" -> "json", ".trec" -> "trec", ".txt" -> "trec". Use the "kind" argument to override this behavior.

        Args:
            path (str): File path.
            kind (str, optional): Kind of file to load, must be either "json" or "trec".

        Returns:
            Qrels: ranx.Qrels
        """
        # Infer file extension -------------------------------------------------
        kind = get_file_kind(path, kind)

        # Load Qrels -----------------------------------------------------------
        if kind == "json":
            qrels = orjson.loads(open(path, "rb").read())
        else:
            qrels = defaultdict(dict)
            with open(path) as f:
                for line in f:
                    q_id, _, doc_id, rel = line.split()
                    qrels[q_id][doc_id] = int(rel)

        return Qrels.from_dict(qrels)

    @staticmethod
    def from_df(
        df: pd.DataFrame,
        q_id_col: str = "q_id",
        doc_id_col: str = "doc_id",
        score_col: str = "score",
    ):
        """Convert a Pandas DataFrame to ranx.Qrels.

        Args:
            df (pd.DataFrame): Qrels as Pandas DataFrame
            q_id_col (str, optional): Query IDs column. Defaults to "q_id".
            doc_id_col (str, optional): Document IDs column. Defaults to "doc_id".
            score_col (str, optional): Relevance score judgments column. Defaults to "score".

        Returns:
            Qrels: ranx.Qrels
        """
        assert (
            df[q_id_col].dtype == "O"
        ), "DataFrame scores column dtype must be `object` (string)"
        assert (
            df[doc_id_col].dtype == "O"
        ), "DataFrame scores column dtype must be `object` (string)"
        assert (
            df[score_col].dtype == int
        ), "DataFrame scores column dtype must be `int`"

        qrels_dict = (
            df.groupby(q_id_col)[[doc_id_col, score_col]]
            .apply(lambda g: {x[0]: x[1] for x in g.values.tolist()})
            .to_dict()
        )

        return Qrels.from_dict(qrels_dict)

    @staticmethod
    def from_ir_datasets(dataset_id: str):
        """Convert `ir-datasets` qrels into ranx.Qrels. It automatically downloads data if missing.
        Args:
            dataset_id (str): ID of the detaset in `ir-datasets`. `ir-datasets` catalog is available here: https://ir-datasets.com/index.html.
        Returns:
            Qrels: ranx.Qrels
        """
        qrels = Qrels.from_dict(ir_datasets.load(dataset_id).qrels_dict())
        qrels.name = dataset_id
        return qrels

    @property
    def size(self):
        return len(self.qrels)

    def __getitem__(self, q_id):
        return dict(self.qrels[q_id])

    def __len__(self) -> int:
        return len(self.qrels)

    def __repr__(self):
        return self.qrels.__repr__()

    def __str__(self):
        return self.qrels.__str__()


def get_file_kind(path: str = "qrels.json", kind: str = None) -> str:
    # Infer file extension
    if kind is None:
        kind = os.path.splitext(path)[1][1:]
        kind = "trec" if kind == "txt" else kind

    # Sanity check
    assert kind in {
        "json",
        "trec",
    }, "Error `kind` must be 'json' or 'trec'"

    return kind
