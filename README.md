# metrics_eval

[metrics_eval](https://github.com/AmenRa/metrics_eval) is a collection of fast metrics implemented in Python, taking advantage of [Numba](https://github.com/numba/numba) for high speed vector operations and automatic parallelization.

The goal of the project is to create a package of high performant metrics implementations that can be used in Python for research.

It currently contains only metrics used for the evaluation of Information Retrieval Systems (Search Engines) and Recommender Systems.

## Available Metrics
### Ranking Metrics:
* Hit List
* Hits
* Precision
* Recall
* rPrecision
* Mean Reciprocal Rank (MRR)
* Average Precision (AP)
* Mean Average Precision (MAP)
* Discounted Cumulative Gain (DCG)
* Ideal Discounted Cumulative Gain (IDCG)
* Normalized Discounted Cumulative Gain (NDCG)

Ranking metrics have been tested against [TREC Eval](https://github.com/usnistgov/trec_eval) — through [pytrec_eval](https://github.com/cvangysel/pytrec_eval) — for correctness.

The implemented metrics are up to more than 50 times faster than [pytrec_eval](https://github.com/cvangysel/pytrec_eval) and with a much lower memory footprint (_see [pytrec_eval_comparison](https://github.com/AmenRa/metrics_eval/tree/master/pytrec_eval_comparison) folder_).

## Requirements
* Numba >= 0.49.1
* Numpy >= 1.15

## Installation
WORK IN PROGRESS

## Usage
```python
from metrics_eval import ranking_metrics as metrics
import numpy as np

y_true = np.array([1, 4, 5, 6])
y_pred = np.array([1, 2, 3, 4, 5, 7])
k = 5

metrics.hits_at_k(y_true, y_pred, k)
>>> 3.0
```

## Documentations
WORK IN PROGRESS

## Citation
WORK IN PROGRESS

## Want to contribute?
WORK IN PROGRESS
