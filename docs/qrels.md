# Qrels

`Qrels`, or _query relevance judgments_, stores the ground truth for conducting evaluations.  
The preferred way for creating a `Qrels` istance is converting Python dictionary as follows:

```python
from ranx import qrels

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
```

Qrels can also be loaded from TREC-style and JSON files, from [ir-datasets](https://ir-datasets.com), and from Pandas DataFrames.

## Load from files
```python
qrels = Qrels.from_file("path/to/json_file")
qrels = Qrels.from_file("path/to/trec_file", kind="trec")
```

## Load from ir-datasets
You can find the full list of the qrels provided by [ir-datasets](https://ir-datasets.com) [here](https://ir-datasets.com).

```python
qrels = Qrels.from_ir_datasets("msmarco-document/dev")
```

## Load from Pandas DataFrames
```python
from pandas import DataFrame

qrels_df = DataFrame.from_dict({
    "q_id":   [ "q_1",  "q_1",  "q_2",  "q_2"  ],
    "doc_id": [ "d_12", "d_25", "d_11", "d_22" ],
    "score":  [  5,      3,      6,      1     ],
})

qrels = Qrels.from_df(
    df=qrels_df,
    q_id_col="q_id",
    doc_id_col="doc_id",
    score_col="score",
)
```