<div align="center">
  <img src="https://repository-images.githubusercontent.com/268892956/750228ec-f3f2-465d-9c17-420c688ba2bc">
</div>

<p align="center">
  <!-- Python -->
  <a href="https://www.python.org" alt="Python">
      <img src="https://badges.aleen42.com/src/python.svg" />
  </a>

  <!-- Docs -->
  <a href="https://rank-eval.readthedocs.io/en/latest/?badge=latest" alt="Documentation Status">
      <img src="https://readthedocs.org/projects/rank-eval/badge/?version=latest" />
  </a>
  <!-- Black -->
  <a href="https://github.com/psf/black" alt="Code style: black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT" alt="License: MIT">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  </a>
  <!-- Google Colab -->
  <a href="https://colab.research.google.com/github/AmenRa/ranx/blob/master/examples/overview.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

## 🔥 News

- [ranx](https://github.com/AmenRa/ranx) will be featured in [ECIR 2022](https://ecir2022.org), the 44th European Conference on Information Retrieval!

## 🤖 Dev Bulletin

- [ranx](https://github.com/AmenRa/ranx) works on [Google Colab](https://colab.research.google.com) now. Unfortunately, Google Colab takes some time to compile the Numba functions the first time you call them...

- If you experienced the `numba.typed` issue, it should now be solved. Please, re-install [ranx](https://github.com/AmenRa/ranx).

## ⚡️ Introduction

[ranx](https://github.com/AmenRa/ranx) is a library of fast ranking evaluation metrics implemented in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), leveraging [Numba](https://github.com/numba/numba) for high-speed vector operations and automatic parallelization. 

It allows you to compare different runs, perform statistical tests, and export a LaTeX table for your scientific publications.

We strongly incourage you to check the example folder to learn how to use [ranx](https://github.com/AmenRa/ranx) in just a few minutes.


## ✨ Available Metrics
* Hits 
* Precision
* Recall
* rPrecision
* Mean Reciprocal Rank (MRR)
* Mean Average Precision (MAP)
* Normalized Discounted Cumulative Gain (NDCG)

The metrics have been tested against [TREC Eval](https://github.com/usnistgov/trec_eval) for correctness.

## 🔌 Installation
```bash
pip install ranx
```

## 💡 Usage

### Create Qrels and Run
```python
from ranx import Qrels, Run, evaluate

qrels = Qrels()
qrels.add_multi(
    q_ids=["q_1", "q_2"],
    doc_ids=[
        ["doc_12", "doc_25"],  # q_1 relevant documents
        ["doc_11", "doc_2"],  # q_2 relevant documents
    ],
    scores=[
        [5, 3],  # q_1 relevance judgements
        [6, 1],  # q_2 relevance judgements
    ],
)

run = Run()
run.add_multi(
    q_ids=["q_1", "q_2"],
    doc_ids=[
        ["doc_12", "doc_23", "doc_25", "doc_36", "doc_32", "doc_35"],
        ["doc_12", "doc_11", "doc_25", "doc_36", "doc_2",  "doc_35"],
    ],
    scores=[
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    ],
)
```

### Evaluate
```python
# Compute score for a single metric
evaluate(qrels, run, "ndcg@5")
>>> 0.7861

# Compute scores for multiple metrics at once
evaluate(qrels, run, ["map@5", "mrr"])
>>> {"map@5": 0.6416, "mrr": 0.75}

# Computed metric scores are saved in the Run object
run.mean_scores
>>> {"ndcg@5": 0.7861, "map@5": 0.6416, "mrr": 0.75}

# Access scores for each query
dict(run.scores)
>>> {"ndcg@5": {"q_1": 0.9430, "q_2": 0.6292},
      "map@5": {"q_1": 0.8333, "q_2": 0.4500},
        "mrr": {"q_1": 1.0000, "q_2": 0.5000}}
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
#    Model    MAP@100     MRR@100     NDCG@10
---  -------  ----------  ----------  ----------
a    model_1  0.3202ᵇ     0.3207ᵇ     0.3684ᵇᶜ
b    model_2  0.2332      0.2339      0.239
c    model_3  0.3082ᵇ     0.3089ᵇ     0.3295ᵇ
d    model_4  0.3664ᵃᵇᶜ   0.3668ᵃᵇᶜ   0.4078ᵃᵇᶜ
e    model_5  0.4053ᵃᵇᶜᵈ  0.4061ᵃᵇᶜᵈ  0.4512ᵃᵇᶜᵈ
```

## 📖 Examples
* [Overview](https://github.com/AmenRa/ranx/tree/master/examples/overview.ipynb): This notebook shows the main features of [ranx](https://github.com/AmenRa/ranx).
* [Create Qrels and Run](https://github.com/AmenRa/ranx/tree/master/examples/create_qrels_and_run.ipynb): This notebook shows different ways of creating `Qrels` and `Run`.

## 📚 Documentation
_To be updated! Please, refer to the examples in the meantime._  
Search the [documentation](https://rank-eval.readthedocs.io/en/latest/) for more details and examples.

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
Would you like to see a new metric implemented? Please, open a [new issue](https://github.com/AmenRa/ranx/issues/new).

## 🤘 Want to contribute?
Would you like to contribute? Please, drop me an [e-mail](mailto:elias.bssn@gmail.com?subject=[GitHub]%20ranx).

## 📄 License

[ranx](https://github.com/AmenRa/ranx) is an open-sourced software licensed under the [MIT license](LICENSE).
