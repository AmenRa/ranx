# Run

`Run` stores the relevance scores estimated by the model under evaluation.<\br>
The preferred way for creating a `Run` istance is converting a Python dictionary as follows:

```python
from ranx import run

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

## From Files
```python
run = Run.from_file("path/to/json_file")
run = Run.from_file("path/to/trec_file", kind="trec")
```

## From Pandas DataFrames
```python
from pandas import DataFrame

run_df = DataFrame.from_dict({
    "q_id":   [ "q_1",  "q_1",  "q_2",  "q_2" ],
    "doc_id": [ "d_12", "d_25", "d_11", "d_22" ],
    "score":  [  5,      3,      6,      1 ],
})

run = Runs.from_df(
    df=run_df,
    q_id_col="q_id",
    doc_id_col="doc_id",
    score_col="score",
)
```