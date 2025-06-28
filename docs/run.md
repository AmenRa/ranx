# Run

`Run` stores the relevance scores estimated by the model under evaluation.
There is no constraint on the score values, i.e., zero and negative scores are not removed. 
The preferred way for creating a `Run` instance is converting a Python dictionary as follows:

```python
from ranx import Run

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
```

`Runs` can also be loaded from TREC-style and JSON files, and from Pandas DataFrames.

## Load from Files
Parse a run file into `ranx.Run`.  
Supported formats are JSON, TREC run, gzipped TREC run, and LZ4.  
Correct import behavior is inferred from the file extension: `.json` -> `json`, `.trec` -> `trec`, `.txt` -> `trec`, `.gz` -> `trec`, `.lz4` -> `lz4`.  
Use the argument `kind` to override the default behavior.
Use the argument `name` to set the name of the run. Default is `None`.

```python
run = Run.from_file("path/to/run.json")  # JSON file
run = Run.from_file("path/to/run.trec")  # TREC-Style file
run = Run.from_file("path/to/run.txt")   # TREC-Style file with txt extension
run = Run.from_file("path/to/run.gz")    # Gzipped TREC-Style file
run = Run.from_file("path/to/run.lz4")   # lz4 file produced by saving a ranx.Run as lz4
run = Run.from_file("path/to/run.custom", kind="json")  # Loaded as JSON file
```

## Load from Pandas DataFrames
`ranx` can load `runs` from Pandas DataFrames.  
The argument `name` is used to set the name of the run. Default is `None`.

```python
from pandas import DataFrame

run_df = DataFrame.from_dict({
    "q_id":   [ "q_1",  "q_1",  "q_2",  "q_2"  ],
    "doc_id": [ "d_12", "d_25", "d_11", "d_22" ],
    "score":  [  0.5,    0.3,    0.6,    0.1   ],
})

run = Run.from_df(
    df=run_df,
    q_id_col="q_id",
    doc_id_col="doc_id",
    score_col="score",
    name="my_run",
)
```

## Load from Parquet files
`ranx` can load `runs` from Parquet files, even from remote sources.  
You can control the behavior of the underlying `pandas.read_parquet` function by passing additional arguments through the `pd_kwargs` argument (see https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html).  
The argument `name` is used to set the name of the run. Default is `None`.

```python
run = Run.from_parquet(
    path="/path/to/parquet/file",
    q_id_col="q_id",
    doc_id_col="doc_id",
    score_col="score",
    pd_kwargs=None,
    name="my_run",
)
```

## Save
Write `run` to `path` as JSON file, TREC run, LZ4 file, or Parquet file.   
File type is automatically inferred form the filename extension: `.json` -> `json`, `.trec` -> `trec`, `.txt` -> `trec`, and `.lz4` -> `lz4`, `.parq` -> `parquet`, `.parquet` -> `parquet`.  
Use the `kind` argument to override this behavior.

```python
run.save("path/to/run.json")     # Save as JSON file
run.save("path/to/run.trec")     # Save as TREC-Style file
run.save("path/to/run.txt")      # Save as TREC-Style file with txt extension
run.save("path/to/run.lz4")      # Save as lz4 file
run.save("path/to/run.parq")     # Save as Parquet file
run.save("path/to/run.parquet")  # Save as Parquet file
run.save("path/to/run.custom", kind="json")  # Save as JSON file
```

## Make comparable
It adds empty results for queries missing from the run and removes those not appearing in qrels.

```python
run.make_comparable(qrels)
```