# metrics_eval

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

[metrics_eval](https://github.com/AmenRa/metrics_eval) is a collection of fast metrics implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), taking advantage of [Numba](https://github.com/numba/numba) for high speed vector operations and automatic parallelization.

The goal of the project is to create a package of high performant metrics implementations that can be used in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) for research.

_It currently contains only metrics for [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval) and [Recommender Systems](https://en.wikipedia.org/wiki/Recommender_system) evaluation._

## ✨ Available Metrics
<details>
  <summary>Ranking Metrics</summary>
  
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

  Ranking metrics have been tested against [TREC Eval](https://github.com/usnistgov/trec_eval) for correctness — through a comparison with [pytrec_eval](https://github.com/cvangysel/pytrec_eval).

  The implemented metrics are up to 50 times faster than [pytrec_eval](https://github.com/cvangysel/pytrec_eval) and with a much lower memory footprint (_see [pytrec_eval_comparison](https://github.com/AmenRa/metrics_eval/tree/master/pytrec_eval_comparison) folder_).
</details>

## 🔧 Requirements
* Python >= 3.7
* Numba >= 0.49.1
* Numpy >= 1.15

## 🔌 Installation
```bash
pip install metrics_eval
```

## 💡 Usage
<details>
  <summary>Examples</summary>
  
  ```python
  from metrics_eval import ndcg
  import numpy as np

  # Note that y_true does not need to be ordered
  # Integers are documents IDs, while floats are the true relevance scores
  y_true = np.array([[[12, 0.5], [25, 0.3]], [[11, 0.4], [2, 0.6]]])
  y_pred = np.array([[12, 234, 25, 36, 32, 35], [12, 11, 25, 36, 2, 35]])
  k = 5

  ndcg(y_true, y_pred, k)
  >>> 0.7525653965843032
  ```

  metrics_eval support the usage of y_true elements of different lenght by using a list of arrays:
  ```python
  from metrics_eval import ndcg
  import numpy as np

  y_true = [np.array([[12, 0.5], [25, 0.3]]), np.array([[11, 0.4], [2, 0.6], [12, 0.1]])]
  y_pred = np.array([[12, 234, 25, 36, 32, 35], [12, 11, 25, 36, 2, 35]])
  k = 5

  ndcg(y_true, y_pred, k)
  >>> 0.7525653965843032
  ```

  However, for maximum speed, consider converting the y_true list to a [Numba Typed List](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#typed-list) by simply using the provided utility function:
  ```python
  from metrics_eval import ndcg, utils
  import numpy as np

  y_true = [np.array([[12, 0.5], [25, 0.3]]), np.array([[11, 0.4], [2, 0.6], [12, 0.1]])]
  y_pred = np.array([[12, 234, 25, 36, 32, 35], [12, 11, 25, 36, 2, 35]])
  k=5

  y_true = utils.to_typed_list(y_true)

  ndcg(y_true, y_pred, k)
  >>> 0.786890544287473
  ```
  
</details>

## 📚 Documentation
Search the [documentation](https://metrics-eval.readthedocs.io/en/latest/) for more details and examples.

## 🎓 Citation
If you end up using [metrics_eval](https://github.com/AmenRa/metrics_eval) to evaluate results for your sceintific publication, please consider citing my work:
```
@misc{metricsEval2020,
  title = {Metrics\_eval: Blazing Fast Evaluation Metrics in Python},
  author = {Bassani, Elias},
  year = {2020},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/AmenRa/metrics_eval}},
}
```

## 🎁 Feature Requests
If you want a metric to be added, please open a [new issue](https://github.com/AmenRa/metrics_eval/issues/new).

## 🤘 Want to contribute?
If you want to contribute, please drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20metrics_eval).

## 📄 License

[metrics_eval](https://github.com/AmenRa/metrics_eval) is an open-sourced software licensed under the [MIT license](LICENSE).
