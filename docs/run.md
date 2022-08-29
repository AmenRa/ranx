# Run

`Run` stores the relevance scores estimated by the model under evaluation.  
The preferred way for creating a `Run` istance is converting a Python dictionary as follows:

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
Supported formats are JSON and TREC run format.  
Correct import behavior is inferred from the file extension: `.json` → `json`, `.trec` → `trec`, `.txt` → `trec`.  
Use the `kind` argument to override the default behavior.

```python
run = Run.from_file("path/to/run.json")  # JSON file
run = Run.from_file("path/to/run.trec")  # TREC-Style file
run = Run.from_file("path/to/run.txt")   # TREC-Style file with txt extension
run = Run.from_file("path/to/run.custom", kind="json")  # Loaded as JSON file
```

## Load from Pandas DataFrames
```python
from pandas import DataFrame

run_df = DataFrame.from_dict({
    "q_id":   [ "q_1",  "q_1",  "q_2",  "q_2"  ],
    "doc_id": [ "d_12", "d_25", "d_11", "d_22" ],
    "score":  [  0.5,    0.3,    0.6,    0.1   ],
})

run = Runs.from_df(
    df=run_df,
    q_id_col="q_id",
    doc_id_col="doc_id",
    score_col="score",
)
```

## Save
Write `run` to `path` as JSON file or TREC run format.  
File type is automatically inferred form the filename extension: `.json` → `json`, `.trec` → `trec`, `.txt` → `trec`.  
Use the `kind` argument to override the default behavior.

```python
run.save("path/to/run.json")  # Save as JSON file
run.save("path/to/run.trec")  # Save as TREC-Style file
run.save("path/to/run.txt")   # Save as TREC-Style file with txt extension
run.save("path/to/run.custom", kind="json")  # Save as JSON file
```