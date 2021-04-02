# rank_eval

<p align="center">
  <!-- Docs -->
  <a href="https://metrics-eval.readthedocs.io/en/latest/?badge=latest" alt="Documentation Status">
      <img src="https://readthedocs.org/projects/metrics-eval/badge/?version=latest" />
  </a>
  <!-- Black -->
  <a href="https://github.com/psf/black" alt="Code style: black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT" alt="License: MIT">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  </a>
</p>

## ⚡️ Introduction

[rank_eval](https://github.com/AmenRa/rank_eval) is a collection of fast ranking evaluation metrics implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), taking advantage of [Numba](https://github.com/numba/numba) for high speed vector operations and automatic parallelization.

## ✨ Available Metrics
* Hits
* Precision
* Recall
* rPrecision
* Mean Reciprocal Rank (MRR)
* Mean Average Precision (MAP)
* Normalized Discounted Cumulative Gain (NDCG)

The metrics have been tested against [TREC Eval](https://github.com/usnistgov/trec_eval) for correctness — through a comparison with [pytrec_eval](https://github.com/cvangysel/pytrec_eval).

The implemented metrics are up to 50 times faster than [pytrec_eval](https://github.com/cvangysel/pytrec_eval) and with a much lower memory footprint.

Please note that `TREC Eval` uses a non-standard NDCG implementation. To mimic its behaviour, pass `trec_eval=True` to `rank_eval`'s `ndcg` function.

## 🔧 Requirements
* Python 3
* Numpy
* Numba

## 🔌 Installation
```bash
pip install rank_eval
```

## 💡 Usage  
```python
from rank_eval import ndcg
import numpy as np

# Note that y_true does not need to be ordered
# Integers are documents IDs, while floats are the true relevance scores
y_true = np.array([[[12, 0.5], [25, 0.3]], [[11, 0.4], [2, 0.6]]])
y_pred = np.array(
    [
        [[12, 0.9], [234, 0.8], [25, 0.7], [36, 0.6], [32, 0.5], [35, 0.4]],
        [[12, 0.9], [11, 0.8], [25, 0.7], [36, 0.6], [2, 0.5], [35, 0.4]],
    ]
)
k = 5

ndcg(y_true, y_pred, k)
>>> 0.7525653965843032
```

rank_eval supports the usage of y_true elements of different lenght by using [Numba Typed List](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#typed-list). Simply convert your y_true list of arrays using the provided utility function:
```python
from rank_eval import ndcg
from rank_eval.utils import to_typed_list
import numpy as np

y_true = [np.array([[12, 0.5], [25, 0.3]]), np.array([[11, 0.4], [2, 0.6], [12, 0.1]])]
y_true = to_typed_list(y_true)
y_pred = np.array(
    [
        [[12, 0.9], [234, 0.8], [25, 0.7], [36, 0.6], [32, 0.5], [35, 0.4]],
        [[12, 0.9], [11, 0.8], [25, 0.7], [36, 0.6], [2, 0.5], [35, 0.4]],
    ]
)
k = 5

ndcg(y_true, y_pred, k)
>>> 0.786890544287473
```

## 📚 Documentation
Search the [documentation](https://rank-eval.readthedocs.io/en/latest/) for more details and examples.

## 🎓 Citation
If you end up using [rank_eval](https://github.com/AmenRa/rank_eval) to evaluate results for your sceintific publication, please consider citing it:
```
@misc{rankEval2021,
  title = {Rank\_eval: Blazing Fast Ranking Evaluation Metrics in Python},
  author = {Bassani, Elias},
  year = {2021},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/AmenRa/rank_eval}},
}
```

## 🎁 Feature Requests
If you want a metric to be added, please open a [new issue](https://github.com/AmenRa/rank_eval/issues/new).

## 🤘 Want to contribute?
If you want to contribute, please drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20rank_eval).

## 📄 License

[rank_eval](https://github.com/AmenRa/rank_eval) is an open-sourced software licensed under the [MIT license](LICENSE).
