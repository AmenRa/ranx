<div align="center">
  <img src="https://repository-images.githubusercontent.com/268892956/750228ec-f3f2-465d-9c17-420c688ba2bc">
</div>

<p align="center">
  <!-- Python -->
  <a href="https://www.python.org" alt="Python">
      <img src="https://badges.aleen42.com/src/python.svg" />
  </a>
  <!-- Version -->
  <!-- <a href="https://pypi.python.org/pypi/ranx"><img src="https://img.shields.io/pypi/v/ranx.svg" alt="PyPI version"></a> -->
  <a href="https://badge.fury.io/py/ranx"><img src="https://badge.fury.io/py/ranx.svg" alt="PyPI version" height="18"></a>
  <!-- Docs -->
  <a href="https://amenra.github.io/ranx"><img src="https://img.shields.io/badge/docs-passing-<COLOR>.svg" alt="Documentation Status"></a>
  <!-- Black -->
  <a href="https://github.com/psf/black" alt="Code style: black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
  </a>
  <!-- License -->
  <a href="https://lbesson.mit-license.org/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <!-- Google Colab -->
  <a href="https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/1_overview.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

## 🔥 News

- [ranx](https://github.com/AmenRa/ranx) will be featured in [ECIR 2022](https://ecir2022.org), the 44th European Conference on Information Retrieval!
- Check out the new [examples](#-examples) on Google Colab! 
- Added a [changelog](changelog) to document few breaking changes introduced in `v.0.1.11`.

## ⚡️ Introduction

[ranx](https://github.com/AmenRa/ranx) is a library of fast ranking evaluation metrics implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), leveraging [Numba](https://github.com/numba/numba) for high-speed vector operations and automatic parallelization.
It offers a user-friendly interface to evaluate and compare [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval) and [Recommender Systems](https://en.wikipedia.org/wiki/Recommender_system).
Moreover, [ranx](https://github.com/AmenRa/ranx) allows you to perform statistical tests and export LaTeX tables for your scientific publications.

For a quick overview, follow the [Usage](#-usage) section.

For a in-depth overview, follow the [Examples](#-examples) section.


## ✨ Available Metrics
* [Hits](https://amenra.github.io/ranx/metrics/#ranx.metrics.hits)
* [Hit Rate](https://amenra.github.io/ranx/metrics/#ranx.metrics.hit_rate)
* [Precision](https://amenra.github.io/ranx/metrics/#ranx.metrics.precision)
* [Recall](https://amenra.github.io/ranx/metrics/#ranx.metrics.recall)
* [F1](https://amenra.github.io/ranx/metrics/#ranx.metrics.f1)
* [r-Precision](https://amenra.github.io/ranx/metrics/#ranx.metrics.r_precision)
* [Mean Reciprocal Rank (MRR)](https://amenra.github.io/ranx/metrics/#ranx.metrics.reciprocal_rank)
* [Mean Average Precision (MAP)](https://amenra.github.io/ranx/metrics/#ranx.metrics.average_precision)
* [Normalized Discounted Cumulative Gain (NDCG)](https://amenra.github.io/ranx/metrics/#ranx.metrics.ndcg)

The metrics have been tested against [TREC Eval](https://github.com/usnistgov/trec_eval) for correctness.

## 🔌 Installation
```bash
pip install ranx
```

## 💡 Usage

### Create Qrels and Run
```python
from ranx import Qrels, Run, evaluate

qrels_dict = { "q_1": { "d_12": 5, "d_25": 3 },
               "q_2": { "d_11": 6, "d_22": 1 } }

run_dict = { "q_1": { "d_12": 0.9, "d_23": 0.8, "d_25": 0.7,
                      "d_36": 0.6, "d_32": 0.5, "d_35": 0.4  },
             "q_2": { "d_12": 0.9, "d_11": 0.8, "d_25": 0.7,
                      "d_36": 0.6, "d_22": 0.5, "d_35": 0.4  } }

qrels = Qrels(qrels_dict)
run = Run(run_dict)
```

### Evaluate
```python
# Compute score for a single metric
evaluate(qrels, run, "ndcg@5")
>>> 0.7861

# Compute scores for multiple metrics at once
evaluate(qrels, run, ["map@5", "mrr"])
>>> {"map@5": 0.6416, "mrr": 0.75}
```

### Compare
```python
# Compare different runs and perform statistical tests
report = compare(
    qrels=qrels,
    runs=[run_1, run_2, run_3, run_4, run_5],
    metrics=["map@100", "mrr@100", "ndcg@10"],
    max_p=0.01  # P-value threshold
)

print(report)
```
Output:
```
>>>
#    Model    MAP@100    MRR@100    NDCG@10
---  -------  --------   --------   ---------
a    model_1  0.320ᵇ     0.320ᵇ     0.368ᵇᶜ
b    model_2  0.233      0.234      0.239
c    model_3  0.308ᵇ     0.309ᵇ     0.330ᵇ
d    model_4  0.366ᵃᵇᶜ   0.367ᵃᵇᶜ   0.408ᵃᵇᶜ
e    model_5  0.405ᵃᵇᶜᵈ  0.406ᵃᵇᶜᵈ  0.451ᵃᵇᶜᵈ
```

## 📖 Examples

| Name                  | Link                                                                                                                                                                                   |
| --------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Overview              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/1_overview.ipynb)              |
| Qrels and Run         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/2_qrels_and_run.ipynb)         |
| Evaluation            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/3_evaluation.ipynb)            |
| Comparison and Report | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/4_comparison_and_report.ipynb) |

## 📚 Documentation
Browse the [documentation](https://amenra.github.io/ranx) for more details and examples.

## 🎓 Citation
If you use [ranx](https://github.com/AmenRa/ranx) to evaluate results for your scientific publication, please consider citing it:
```
@misc{ranx2021,
  title = {ranx: A Blazing-Fast Python Library for Ranking Evaluation and Comparison},
  author = {Bassani, Elias},
  year = {2021},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/AmenRa/ranx}},
}
```

## 🎁 Feature Requests
Would you like to see other features implemented? Please, open a [feature request](https://github.com/AmenRa/ranx/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFeature+Request%5D+title).

## 🤘 Want to contribute?
Would you like to contribute? Please, drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20ranx).

## 📄 License

[ranx](https://github.com/AmenRa/ranx) is an open-sourced software licensed under the [MIT license](LICENSE).
