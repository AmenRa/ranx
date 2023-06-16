<div align="center">
  <img src="https://repository-images.githubusercontent.com/268892956/750228ec-f3f2-465d-9c17-420c688ba2bc">
</div>

<p align="center">
  <!-- Python -->
  <a href="https://www.python.org" alt="Python"><img src="https://badges.aleen42.com/src/python.svg"></a>
  <!-- Version -->
  <a href="https://pypi.org/project/ranx/"><img src="https://img.shields.io/pypi/v/ranx?color=light-green" alt="PyPI version"></a>
  <!-- Downloads -->
  <a href="https://pepy.tech/project/ranx"><img src="https://static.pepy.tech/personalized-badge/ranx?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads" alt="Download counter"></a>
  <!-- Docs -->
  <a href="https://amenra.github.io/ranx"><img src="https://img.shields.io/badge/docs-passing-<COLOR>.svg" alt="Documentation Status"></a>
  <!-- Black -->
  <a href="https://github.com/psf/black" alt="Code style: black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <!-- License -->
  <a href="https://lbesson.mit-license.org/"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <!-- Google Colab -->
  <a href="https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/1_overview.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>
</p>

## 🔥 News

- 📌 [April 4, 2023] [ranxhub](https://amenra.github.io/ranxhub), the [ranx](https://github.com/AmenRa/ranx)'s companion repository, will be featured in [SIGIR 2023](https://sigir.org/sigir2023)!  
On [ranxhub](https://amenra.github.io/ranxhub), you can download and share pre-computed runs for Information Retrieval datasets, such as [MSMARCO Passage Ranking](https://arxiv.org/abs/1611.09268).

- [June 16 2023] `ranx` `0.3.13` is out!  
This release exposes `DCG` among the available metrics.

- [May 1 2023] `ranx` `0.3.8` is out!  
This release adds early support for results plotting. Specifically, it is now possible to plot Interpolated Precision-Recall Curve. Click [here](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/7_plot.ipynb) for further details.
<!-- This release adds support for changing Qrels relevance level, i.e, the minimum relevance judgement score to consider a document to be relevant.   -->
<!-- You can now define metric-wise relevance levels by appending `-l<num>` to metric names (e.g., `evaluate(qrels, run, ["map@100-l2", "ndcg-l3])`), or setting the Qrels relevance level qrels-wise as `qrels.set_relevance_level(2)`. -->

## ⚡️ Introduction

[ranx](https://github.com/AmenRa/ranx) ([raŋks]) is a library of fast ranking evaluation metrics implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), leveraging [Numba](https://github.com/numba/numba) for high-speed [vector operations](https://en.wikipedia.org/wiki/Automatic_vectorization) and [automatic parallelization](https://en.wikipedia.org/wiki/Automatic_parallelization).
It offers a user-friendly interface to evaluate and compare [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval) and [Recommender Systems](https://en.wikipedia.org/wiki/Recommender_system).
[ranx](https://github.com/AmenRa/ranx) allows you to perform statistical tests and export [LaTeX](https://en.wikipedia.org/wiki/LaTeX) tables for your scientific publications.
Moreover, [ranx](https://github.com/AmenRa/ranx) provides several [fusion algorithms](https://amenra.github.io/ranx/fusion) and [normalization strategies](https://amenra.github.io/ranx/normalization), and an automatic [fusion optimization](https://amenra.github.io/ranx/fusion/#optimize-fusion) functionality.
[ranx](https://github.com/AmenRa/ranx) was featured in [ECIR 2022](https://ecir2022.org) and [CIKM 2022](https://www.cikm2022.org). 
 
If you use [ranx](https://github.com/AmenRa/ranx) to evaluate results or conducting experiments involving fusion for your scientific publication, please consider citing it: [evaluation bibtex](https://dblp.org/rec/conf/ecir/Bassani22.html?view=bibtex), [fusion bibtex](https://dblp.org/rec/conf/cikm/BassaniR22.html?view=bibtex).

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
* [Bpref](https://amenra.github.io/ranx/metrics/#bpref)
* [Rank-biased Precision (RBP)](https://amenra.github.io/ranx/metrics/#rank-biased-precision)
* [Mean Reciprocal Rank (MRR)](https://amenra.github.io/ranx/metrics/#mean-reciprocal-rank)
* [Mean Average Precision (MAP)](https://amenra.github.io/ranx/metrics/#mean-average-precision)
* [Discounted Cumulative Gain (DCG)](https://amenra.github.io/ranx/metrics/#dcg)
* [Normalized Discounted Cumulative Gain (NDCG)](https://amenra.github.io/ranx/metrics/#ndcg)

The metrics have been tested against [TREC Eval](https://github.com/usnistgov/trec_eval) for correctness.

### Statistical Tests
* [Paired Student's t-Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/t_test.htm) (default)
* [Fisher's Randomization Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/fishrand.htm)
* [Tukey's HSD Test](https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm)

Please, refer to [Smucker et al.](https://dl.acm.org/doi/10.1145/1321440.1321528), [Carterette](https://dl.acm.org/doi/10.1145/2094072.2094076), and [Fuhr](http://www.sigir.org/wp-content/uploads/2018/01/p032.pdf) for additional information on statistical tests for Information Retrieval.

### Off-the-shelf Qrels
You can load qrels from [ir-datasets](https://ir-datasets.com) as simply as:
```python
qrels = Qrels.from_ir_datasets("msmarco-document/dev")
```
A full list of the available qrels is provided [here](https://ir-datasets.com).

### Off-the-shelf Runs
You can load runs from [ranxhub](https://amenra.github.io/ranxhub/) as simply as:
```python
run = Run.from_ranxhub("run-id")
```
A full list of the available runs is provided [here](https://amenra.github.io/ranxhub//browse).

### Fusion Algorithms

| **Name**                                                 | **Name**                                                   | **Name**                                                                | **Name**                                                     | **Name**                                                                       |
| -------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| [CombMIN](https://amenra.github.io/ranx/fusion/#combmin) | [CombMNZ](https://amenra.github.io/ranx/fusion/#combmnz)   | [RRF](https://amenra.github.io/ranx/fusion/#reciprocal-rank-fusion-rrf) | [MAPFuse](https://amenra.github.io/ranx/fusion/#mapfuse)     | [BordaFuse](https://amenra.github.io/ranx/fusion/#bordafuse)                   |
| [CombMED](https://amenra.github.io/ranx/fusion/#combmed) | [CombGMNZ](https://amenra.github.io/ranx/fusion/#combgmnz) | [RBC](https://amenra.github.io/ranx/fusion/#rank-biased-centroids-rbc)  | [PosFuse](https://amenra.github.io/ranx/fusion/#posfuse)     | [Weighted BordaFuse](https://amenra.github.io/ranx/fusion/#weighted-bordafuse) |
| [CombANZ](https://amenra.github.io/ranx/fusion/#combanz) | [ISR](https://amenra.github.io/ranx/fusion/#isr)           | [WMNZ](https://amenra.github.io/ranx/fusion/#wmnz)                      | [ProbFuse](https://amenra.github.io/ranx/fusion/#probfuse)   | [Condorcet](https://amenra.github.io/ranx/fusion/#condorcet)                   |
| [CombMAX](https://amenra.github.io/ranx/fusion/#combmax) | [Log_ISR](https://amenra.github.io/ranx/fusion/#log_isr)   | [Mixed](https://amenra.github.io/ranx/fusion/#mixed)                    | [SegFuse](https://amenra.github.io/ranx/fusion/#segfuse)     | [Weighted Condorcet](https://amenra.github.io/ranx/fusion/#weighted-condorcet) |
| [CombSUM](https://amenra.github.io/ranx/fusion/#combsum) | [LogN_ISR](https://amenra.github.io/ranx/fusion/#logn_isr) | [BayesFuse](https://amenra.github.io/ranx/fusion/#bayesfuse)            | [SlideFuse](https://amenra.github.io/ranx/fusion/#slidefuse) | [Weighted Sum](https://amenra.github.io/ranx/fusion/#wighted-sum)              |

Please, refer to the [documentation](https://amenra.github.io/ranx/fusion) for further details.

### Normalization Strategies

* [Min-Max Norm](https://amenra.github.io/ranx/normalization/#min-max-norm) 
* [Max Norm](https://amenra.github.io/ranx/normalization/#sum-norm)         
* [Sum Norm](https://amenra.github.io/ranx/normalization/#rank-norm)        
* [ZMUV Norm](https://amenra.github.io/ranx/normalization/#max-norm)   
* [Rank Norm](https://amenra.github.io/ranx/normalization/#zmuv-norm)  
* [Borda Norm](https://amenra.github.io/ranx/normalization/#borda-norm)

Please, refer to the [documentation](https://amenra.github.io/ranx/fusion) for further details.



## 🔌 Requirements
```bash
python>=3.8
```
As of `v.0.3.5`, [ranx](https://github.com/AmenRa/ranx) requires `python>=3.8`.

## 💾 Installation 

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

# Compare different runs and perform Two-sided Paired Student's t-Test
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

| Name                                                             | Link                                                                                                                                                                                   |
| ---------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Overview                                                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/1_overview.ipynb)              |
| Qrels and Run                                                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/2_qrels_and_run.ipynb)         |
| Evaluation                                                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/3_evaluation.ipynb)            |
| Comparison and Report                                            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/4_comparison_and_report.ipynb) |
| Fusion                                                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/5_fusion.ipynb)                |
| Plot                                                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/7_plot.ipynb)                  |
| Share your runs with [ranxhub](https://amenra.github.io/ranxhub) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/6_ranxhub.ipynb)               |


## 📚 Documentation
Browse the [documentation](https://amenra.github.io/ranx) for more details and examples.


## 🎓 Citation
If you use [ranx](https://github.com/AmenRa/ranx) to evaluate results for your scientific publication, please consider citing our [ECIR 2022](https://ecir2022.org) paper:
<details>
  <summary>BibTeX</summary>
  
  ```bibtex
  @inproceedings{DBLP:conf/ecir/Bassani22,
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
</details>  

If you use the fusion functionalities provided by [ranx](https://github.com/AmenRa/ranx) for conducting the experiments of your scientific publication, please consider citing our [CIKM 2022](https://www.cikm2022.org) paper:
<details>
  <summary>BibTeX</summary>
  
  ```bibtex
  @inproceedings{DBLP:conf/cikm/BassaniR22,
    author    = {Elias Bassani and
                Luca Romelli},
    title     = {ranx.fuse: {A} Python Library for Metasearch},
    booktitle = {{CIKM}},
    pages     = {4808--4812},
    publisher = {{ACM}},
    year      = {2022}
  }
  ```
</details>

## 🎁 Feature Requests
Would you like to see other features implemented? Please, open a [feature request](https://github.com/AmenRa/ranx/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFeature+Request%5D+title).


## 🤘 Want to contribute?
Would you like to contribute? Please, drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20ranx).


## 📄 License
[ranx](https://github.com/AmenRa/ranx) is an open-sourced software licensed under the [MIT license](LICENSE).
