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

- [June 9, 2022] Added support for **24 fusion algorithms**, **six normalization strategies**, and an **automatic fusion optimization** functionality in `v.0.2`.  
Check out the [official documentation](https://amenra.github.io/ranx) for further details on [fusion](https://amenra.github.io/ranx/fusion) and [normalization](https://amenra.github.io/ranx/normalization).
- [May 18, 2022] Added support for loading qrels from [ir-datasets](https://ir-datasets.com) in `v.0.1.13`.  
Usage example: `Qrels.from_ir_datasets("msmarco-document/dev")` for [MS MARCO](https://microsoft.github.io/msmarco/) document retrieval dev set.
- [May 4, 2022] Added [Paired Student's t-Test](https://en.wikipedia.org/wiki/Student%27s_t-test) in `v.0.1.12`.

## ⚡️ Introduction

[ranx](https://github.com/AmenRa/ranx) is a library of fast ranking evaluation metrics implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), leveraging [Numba](https://github.com/numba/numba) for high-speed [vector operations](https://en.wikipedia.org/wiki/Automatic_vectorization) and [automatic parallelization](https://en.wikipedia.org/wiki/Automatic_parallelization).
It offers a user-friendly interface to evaluate and compare [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval) and [Recommender Systems](https://en.wikipedia.org/wiki/Recommender_system).
[ranx](https://github.com/AmenRa/ranx) allows you to perform statistical tests and export [LaTeX](https://en.wikipedia.org/wiki/LaTeX) tables for your scientific publications.
Moreover, [ranx](https://github.com/AmenRa/ranx) provides several [fusion algorithms](https://amenra.github.io/ranx/fusion) and [normalization strategies](https://amenra.github.io/ranx/normalization), and an automatic [fusion optimization](https://amenra.github.io/ranx/fusion/#optimize-fusion) functionality.
[ranx](https://github.com/AmenRa/ranx) was featured in [ECIR 2022](https://ecir2022.org), the 44th European Conference on Information Retrieval. 
 
If you use [ranx](https://github.com/AmenRa/ranx) to evaluate results or conducting experiments involving fusion for your scientific publication, please consider [citing it](https://dblp.org/rec/conf/ecir/Bassani22.html?view=bibtex).

For a quick overview, follow the [Usage](#-usage) section.

For a in-depth overview, follow the [Examples](#-examples) section.


## ✨ Features

### Metrics
* [Hits](https://amenra.github.io/ranx/metrics/#hits)
* [Hit Rate](https://amenra.github.io/ranx/metrics/#hit-rate-success)
* [Precision](https://amenra.github.io/ranx/metrics/#precision)
* [Recall](https://amenra.github.io/ranx/metrics/#recall)
* [F1](https://amenra.github.io/ranx/metrics/#f1)
* [r-Precision](https://amenra.github.io/ranx/metrics/#r-precision)
* [Mean Reciprocal Rank (MRR)](https://amenra.github.io/ranx/metrics/#mean-reciprocal-rank)
* [Mean Average Precision (MAP)](https://amenra.github.io/ranx/metrics/#mean-average-precision)
* [Normalized Discounted Cumulative Gain (NDCG)](https://amenra.github.io/ranx/metrics/#ndcg)

The metrics have been tested against [TREC Eval](https://github.com/usnistgov/trec_eval) for correctness.

### Statistical Tests
* [Fisher's Randomization Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/fishrand.htm)
* [Paired Student's t-Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/t_test.htm)

Please, refer to [Smucker et al.](https://dl.acm.org/doi/10.1145/1321440.1321528) for additional information on statistical tests for Information Retrieval.

### Off-the-shelf Qrels
You can load qrels from [ir-datasets](https://ir-datasets.com) as simply as:
```python
qrels = Qrels.from_ir_datasets("msmarco-document/dev")
```
A full list of the available qrels is provided [here](https://ir-datasets.com).

### Fusion Algorithms

| **Name**                                                | **Name**                                                               | **Name**                                                    | **Name**                                                                      |
| ------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [CombMIN](https://amenra.github.io/ranx/fusion/#combmin) | [CombGMNZ](https://amenra.github.io/ranx/fusion/#combgmnz)              | [PosFuse](https://amenra.github.io/ranx/fusion/#posfuse)     | [BordaFuse](https://amenra.github.io/ranx/fusion/#bordafuse)                   |
| [CombMED](https://amenra.github.io/ranx/fusion/#combmed) | [ISR](https://amenra.github.io/ranx/fusion/#isr)                        | [ProbFuse](https://amenra.github.io/ranx/fusion/#probfuse)   | [Weighted BordaFuse](https://amenra.github.io/ranx/fusion/#weighted-bordafuse) |
| [CombANZ](https://amenra.github.io/ranx/fusion/#combanz) | [Log_ISR](https://amenra.github.io/ranx/fusion/#log_isr)                | [SegFuse](https://amenra.github.io/ranx/fusion/#segfuse)     | [Condorcet](https://amenra.github.io/ranx/fusion/#condorcet)                   |
| [CombMAX](https://amenra.github.io/ranx/fusion/#combmax) | [LogN_ISR](https://amenra.github.io/ranx/fusion/#logn_isr)              | [SlideFuse](https://amenra.github.io/ranx/fusion/#slidefuse) | [Weighted Condorcet](https://amenra.github.io/ranx/fusion/#weighted-condorcet) |
| [CombSUM](https://amenra.github.io/ranx/fusion/#combsum) | [RRF](https://amenra.github.io/ranx/fusion/#reciprocal-rank-fusion-rrf) | [MAPFuse](https://amenra.github.io/ranx/fusion/#mapfuse)     | [Mixed](https://amenra.github.io/ranx/fusion/#mixed)                           |
| [CombMNZ](https://amenra.github.io/ranx/fusion/#combmnz) | [WMNZ](https://amenra.github.io/ranx/fusion/#wmnz)                      | [BayesFuse](https://amenra.github.io/ranx/fusion/#bayesfuse) | [Wighted Sum](https://amenra.github.io/ranx/fusion/#wighted-sum)               |

Please, refer to the [documentation](https://amenra.github.io/ranx/fusion) for further details.

### Normalization Strategies

* [Min-Max Norm](https://amenra.github.io/ranx/normalization/#min-max-norm) 
* [Max Norm](https://amenra.github.io/ranx/normalization/#sum-norm)         
* [Sum Norm](https://amenra.github.io/ranx/normalization/#rank-norm)        
* [ZMUV Norm](https://amenra.github.io/ranx/normalization/#max-norm)   
* [Rank Norm](https://amenra.github.io/ranx/normalization/#zmuv-norm)  
* [Borda Norm](https://amenra.github.io/ranx/normalization/#borda-norm)

Please, refer to the [documentation](https://amenra.github.io/ranx/fusion) for further details.


## 🔌 Installation
```bash
pip install ranx
```

## 💡 Usage

### Create Qrels and Run
```python
from ranx import Qrels, Run

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
from ranx import evaluate

# Compute score for a single metric
evaluate(qrels, run, "ndcg@5")
>>> 0.7861

# Compute scores for multiple metrics at once
evaluate(qrels, run, ["map@5", "mrr"])
>>> {"map@5": 0.6416, "mrr": 0.75}
```

### Compare
```python
from ranx import compare

# Compare different runs and perform statistical tests
report = compare(
    qrels=qrels,
    runs=[run_1, run_2, run_3, run_4, run_5],
    metrics=["map@100", "mrr@100", "ndcg@10"],
    max_p=0.01  # P-value threshold
)
```
Output:
```python
print(report)
```
```
#    Model    MAP@100    MRR@100    NDCG@10
---  -------  --------   --------   ---------
a    model_1  0.320ᵇ     0.320ᵇ     0.368ᵇᶜ
b    model_2  0.233      0.234      0.239
c    model_3  0.308ᵇ     0.309ᵇ     0.330ᵇ
d    model_4  0.366ᵃᵇᶜ   0.367ᵃᵇᶜ   0.408ᵃᵇᶜ
e    model_5  0.405ᵃᵇᶜᵈ  0.406ᵃᵇᶜᵈ  0.451ᵃᵇᶜᵈ
```

### Fusion
```python
from ranx import fuse, optimize_fusion

best_params = optimize_fusion(
    qrels=train_qrels,
    runs=[train_run_1, train_run_2, train_run_3],
    norm="min-max",     # The norm. to apply before fusion
    method="wsum",      # The fusion algorithm to use (Weighted Sum)
    metric="ndcg@100",  # The metric to maximize
)

combined_test_run = fuse(
    runs=[test_run_1, test_run_2, test_run_3],  
    norm="min-max",       
    method="wsum",        
    params=best_params,
)
```

## 📖 Examples

| Name                  | Link                                                                                                                                                                                   |
| --------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Overview              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/1_overview.ipynb)              |
| Qrels and Run         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/2_qrels_and_run.ipynb)         |
| Evaluation            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/3_evaluation.ipynb)            |
| Comparison and Report | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/4_comparison_and_report.ipynb) |
| Fusion                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/5_fusion.ipynb)                |


## 📚 Documentation
Browse the [documentation](https://amenra.github.io/ranx) for more details and examples.


## 🎓 Citation
If you use [ranx](https://github.com/AmenRa/ranx) to evaluate results for your scientific publication, please consider citing it:
```
@inproceedings{bassani2022ranx,
  author    = {Elias Bassani},
  title     = {ranx: {A} Blazing-Fast Python Library for Ranking Evaluation and Comparison},
  booktitle = {{ECIR} {(2)}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13186},
  pages     = {259--264},
  publisher = {Springer},
  year      = {2022}
}
```


## 🎁 Feature Requests
Would you like to see other features implemented? Please, open a [feature request](https://github.com/AmenRa/ranx/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFeature+Request%5D+title).


## 🤘 Want to contribute?
Would you like to contribute? Please, drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20ranx).


## 📄 License
[ranx](https://github.com/AmenRa/ranx) is an open-sourced software licensed under the [MIT license](LICENSE).
