import gzip
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numba import types
from numba.typed import Dict as TypedDict
from numba.typed import List as TypedList

from ..io import download, load_json, load_lz4, save_json, save_lz4
from .common import (
    add_and_sort,
    create_and_sort,
    sort_dict_by_key,
    sort_dict_of_dict_by_value,
    to_typed_list,
)
from .generic import create_empty_results_dict
from .qrels import Qrels


class Run(object):
    """`Run` stores the relevance scores estimated by the model under evaluation.<\br>
    The preferred way for creating a `Run` instance is converting a Python dictionary as follows:

    ```python
    run_dict = {
        "q_1": {
            "d_1": 1.5,
            "d_2": 2.6,
        },
        "q_2": {
            "d_3": 2.8,
            "d_2": 1.2,
            "d_5": 3.1,
        },
    }

    run = Run(run_dict, name="bm25")

    run = Run()  # Creates an empty Run with no name
    ```
    """
    metadata: Dict[str, Any]
    scores: Dict[str, Dict[str, float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]

    def __init__(self, run: Union[None, Dict[str, Dict[str, float]]] = None, name: Union[None, str] = None):
        """Initialize a Run object.

        Args:
            run (Dict[str, Dict[str, float]], optional): A dictionary mapping query IDs to dictionaries of document scores.
                The inner dictionaries map document IDs to their relevance scores.
                Example:
                ```python
                {
                    "q_1": {
                        "d_1": 1.5,
                        "d_2": 2.6,
                    },
                    "q_2": {
                        "d_3": 2.8,
                        "d_2": 1.2,
                        "d_5": 3.1,
                    },
                }
                ```
                If None, creates an empty Run. Defaults to None.
            name (str, optional): A name for the run (e.g., "bm25", "t5", etc.). Defaults to None.

        Note:
            - Query IDs and document IDs can be any string
            - Scores must be numeric values
            - The run will be automatically sorted by score in descending order
        """
        if run is None:
            self.run = TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.DictType(types.unicode_type, types.float64),
            )
            self.sorted = False
        else:
            # Query IDs
            q_ids = list(run.keys())
            q_ids = TypedList(q_ids)

            # Doc IDs
            doc_ids = [list(doc.keys()) for doc in run.values()]
            max_len = max(len(y) for x in doc_ids for y in x)
            dtype = f"<U{max_len}"
            doc_ids = TypedList([np.array(x, dtype=dtype) for x in doc_ids])

            # Scores
            scores = [list(doc.values()) for doc in run.values()]
            scores = TypedList([np.array(x, dtype=float) for x in scores])
            self.run = create_and_sort(q_ids, doc_ids, scores)
            self.sorted = True

        self.name = name
        self.metadata = {}
        self.scores = defaultdict(dict)
        self.mean_scores = {}
        self.std_scores = {}

    def keys(self):
        """Returns query ids. Used internally."""
        return self.run.keys()

    def add_score(self, q_id: str, doc_id: str, score: int):
        """Add a (doc_id, score) pair to a query (or, change its value if it already exists).

        Args:
            q_id (str): Query ID
            doc_id (str): Document ID
            score (int): Relevance score
        """
        if self.run.get(q_id) is None:
            self.run[q_id] = TypedDict.empty(
                key_type=types.unicode_type,
                value_type=types.float64,
            )
        self.run[q_id][doc_id] = float(score)
        self.sorted = False

    def add(self, q_id: str, doc_ids: List[str], scores: List[float]):
        """Add a query and its relevant documents with the associated relevance score.

        Args:
            q_id (str): Query ID
            doc_ids (List[str]): List of Document IDs
            scores (List[int]): List of relevance scores
        """
        self.add_multi([q_id], [doc_ids], [scores])

    def add_multi(
        self,
        q_ids: List[str],
        doc_ids: List[List[str]],
        scores: List[List[float]],
    ):
        """Add multiple queries at once.

        Args:
            q_ids (List[str]): List of Query IDs
            doc_ids (List[List[str]]): List of list of Document IDs
            scores (List[List[int]]): List of list of relevance scores
        """
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

    def make_comparable(self, qrels: Qrels):
        """Adds empty results for queries missing from the run and removes those not appearing in qrels."""
        # Adds empty results for missing queries
        for q_id in qrels.qrels:
            if q_id not in self.run:
                self.run[q_id] = create_empty_results_dict()

        # Remove results for additional queries
        for q_id in self.run:
            if q_id not in qrels.qrels:
                del self.run[q_id]

        self.sort()

        return self

    def to_typed_list(self):
        """Convert Run to Numba Typed List. Used internally."""
        if not self.sorted:
            self.sort()
        return to_typed_list(self.run)

    def to_dict(self):
        """Convert Run to Python dictionary.

        Returns:
            Dict[str, Dict[str, int]]: Run as Python dictionary
        """
        d = defaultdict(dict)
        for q_id in self.keys():
            d[q_id] = dict(self[q_id])
        return d

    def to_dataframe(self) -> pd.DataFrame:
        """Convert Run to Pandas DataFrame with the following columns: `q_id`, `doc_id`, and `score`.

        Returns:
            pandas.DataFrame: Run as Pandas DataFrame.
        """
        data = {"q_id": [], "doc_id": [], "score": []}

        for q_id in self.run:
            for doc_id in self.run[q_id]:
                data["q_id"].append(q_id)
                data["doc_id"].append(doc_id)
                data["score"].append(self.run[q_id][doc_id])

        return pd.DataFrame.from_dict(data)

    def save(self, path: str = "run.json", kind: str = None):
        """Write `run` to `path` as JSON file, TREC run, LZ4 file, or Parquet file. File type is automatically inferred form the filename extension: ".json" -> "json", ".trec" -> "trec", ".txt" -> "trec", and ".lz4" -> "lz4", ".parq" -> "parquet", ".parquet" -> "parquet". Use the "kind" argument to override this behavior.

        Args:
            path (str, optional): Saving path. Defaults to "run.json".
            kind (str, optional): Kind of file to save, must be either "json", "trec", or "ranxhub". If None, it will be automatically inferred from the filename extension.
        """
        # Infer file extension -------------------------------------------------
        kind = get_file_kind(path, kind)

        # Save Run -------------------------------------------------------------
        if not self.sorted:
            self.sort()

        if kind == "json":
            save_json(self.to_dict(), path)
        elif kind == "lz4":
            save_lz4(self.to_dict(), path)
        elif kind == "parquet":
            self.to_dataframe().to_parquet(path, index=False)
        else:
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
    def from_dict(d: Dict[str, Dict[str, float]], name: str = None):
        """Convert a Python dictionary in form of {q_id: {doc_id: score}} to ranx.Run.

        Args:
            d (Dict[str, Dict[str, int]]): Run as Python dictionary
            name (str, optional): Run name. Defaults to None.

        Returns:
            Run: ranx.Run
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
        scores = TypedList([np.array(x, dtype=float) for x in scores])

        run = Run()
        run.run = create_and_sort(q_ids, doc_ids, scores)
        run.sorted = True
        run.name = name

        return run

    @staticmethod
    def from_file(path: str, kind: str = None, name: str = None):
        """Parse a run file into ranx.Run. Supported formats are JSON, TREC run, gzipped TREC run, and LZ4. Correct import behavior is inferred from the file extension: ".json" -> "json", ".trec" -> "trec", ".txt" -> "trec", ".gz" -> "gzipped trec", ".lz4" -> "lz4". Use the "kind" argument to override this behavior.

        Args:
            path (str): File path.
            kind (str, optional): Kind of file to load, must be either "json" or "trec".
            name (str, optional): Run name. Defaults to None.

        Returns:
            Run: ranx.Run
        """
        # Infer file extension -------------------------------------------------
        kind = get_file_kind(path, kind)

        # Load Run -------------------------------------------------------------
        if kind == "json":
            run = load_json(path)
        elif kind == "lz4":
            run = load_lz4(path)
        else:
            run = defaultdict(dict)
            with gzip.open(path, "rt") if kind == "gz" else open(path) as f:
                for line in f:
                    q_id, _, doc_id, _, rel, run_name = line.split()
                    run[q_id][doc_id] = float(rel)
                    if name is None:
                        name = run_name

        run = Run.from_dict(run, name)

        return run

    @staticmethod
    def from_df(
        df: pd.DataFrame,
        q_id_col: str = "q_id",
        doc_id_col: str = "doc_id",
        score_col: str = "score",
        name: str = None,
    ):
        """Convert a Pandas DataFrame to ranx.Run.

        Args:
            df (pd.DataFrame): Run as Pandas DataFrame
            q_id_col (str, optional): Query IDs column. Defaults to "q_id".
            doc_id_col (str, optional): Document IDs column. Defaults to "doc_id".
            score_col (str, optional): Relevance scores column. Defaults to "score".
            name (str, optional): Run name. Defaults to None.

        Returns:
            Run: ranx.Run
        """
        assert (
            df[q_id_col].dtype == "O"
        ), "DataFrame Query IDs column dtype must be `object` (string)"
        assert (
            df[doc_id_col].dtype == "O"
        ), "DataFrame Document IDs column dtype must be `object` (string)"
        assert (
            df[score_col].dtype == np.float64
        ), "DataFrame scores column dtype must be `float`"

        run_py = (
            df.groupby(q_id_col)[[doc_id_col, score_col]]
            .apply(lambda g: {x[0]: x[1] for x in g.values.tolist()})
            .to_dict()
        )

        return Run.from_dict(run_py, name)

    @staticmethod
    def from_parquet(
        path: str,
        q_id_col: str = "q_id",
        doc_id_col: str = "doc_id",
        score_col: str = "score",
        pd_kwargs: Dict[str, Any] = None,
        name: str = None,
    ):
        """Convert a Parquet file to ranx.Run.

        Args:
            path (str): File path.
            q_id_col (str, optional): Query IDs column. Defaults to "q_id".
            doc_id_col (str, optional): Document IDs column. Defaults to "doc_id".
            score_col (str, optional): Relevance scores column. Defaults to "score".
            pd_kwargs (Dict[str, Any], optional): Additional arguments to pass to `pandas.read_parquet` (see https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html). Defaults to None.
            name (str, optional): Run name. Defaults to None.

        Returns:
            Run: ranx.Run
        """
        pd_kwargs = {} if pd_kwargs is None else pd_kwargs

        return Run.from_df(
            df=pd.read_parquet(path, *pd_kwargs),
            q_id_col=q_id_col,
            doc_id_col=doc_id_col,
            score_col=score_col,
            name=name,
        )

    @staticmethod
    def from_ranxhub(id: str):
        """Download and load a ranx.Run from ranxhub.

        Args:
            path (str): Run ID.

        Returns:
            Run: ranx.Run
        """
        content = download(id)

        run = Run.from_dict(content["run"])
        run.name = content["metadata"]["run"]["name"]
        run.metadata = content["metadata"]

        return run

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


def get_file_kind(path: str = "run.json", kind: str = None) -> str:
    # Infer file extension
    if kind is None:
        kind = os.path.splitext(path)[1][1:]
        kind = "trec" if kind == "txt" else kind
        kind = "parquet" if kind == "parq" else kind

    # Sanity check
    assert kind in {
        "json",
        "trec",
        "lz4",
        "gz",
        "parquet",
    }, "Error `kind` must be 'json', 'trec', 'lz4', 'gz', or 'parquet'"

    return kind
