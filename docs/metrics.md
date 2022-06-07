# Metrics

## Aliases

Aliases to use with `ranx.evaluate` and `ranx.compare`.

| **Metric**                                       | **Alias**   | **@k** |
| ------------------------------------------------ | ----------- | :----: |
| [Hits][hits]                                     | hits        |  yes   |
| [Hit Rate / Success][hit-rate-success]           | hit_rate    |  yes   |
| [Precision][precision]                           | precision   |  yes   |
| [Recall][recall]                                 | recall      |  yes   |
| [F1][f1]                                         | f1          |  yes   |
| [R-Precision][r-precision]                       | r_precision |   no   |
| [Mean Reciprocal Rank][mean-reciprocal-rank]     | mrr         |  yes   |
| [Mean Average Precision][mean-average-precision] | map         |  yes   |
| [NDCG][ndcg]                                     | ndcg        |  yes   |
| [NDCG Burges][ndcg-burges]                       | ndcg_burges |  yes   |

## Hits
**Hits** is the number of relevant documents retrieved.

## Hit Rate / Success
**Hit Rate** is the fraction of queries for which at least one relevant document is retrieved.
Note: it is equivalent to `success` from [trec_eval](https://github.com/usnistgov/trec_eval).

## Precision
---
**Precision** is the proportion of the retrieved documents that are relevant.

$$
\operatorname{Precision}=\frac{r}{n}
$$

where,

- $r$ is the number of retrieved relevant documents;
- $n$ is the number of retrieved documents.

## Recall
---
**Recall** is the ratio between the retrieved documents that are relevant and the total number of relevant documents.

$$
\operatorname{Recall}=\frac{r}{R}
$$

where,

- $r$ is the number of retrieved relevant documents;
- $R$ is the total number of relevant documents.

## F1
---
**F1** is the harmonic mean of [**Precision**][precision] and [**Recall**][recall].

$$
\operatorname{F1} = 2 \times \frac{\operatorname{Precision} \times \operatorname{Recall}}{\operatorname{Precision} + \operatorname{Recall}}
$$

## R-Precision
---
For a given query $Q$, **R-Precision** is the precision at $R$, where $R$ is the number of relevant documents for $Q$. In other words, if there are $r$ relevant documents among the top-$R$ retrieved documents, then R-precision is:

$$
\operatorname{R-Precision} = \frac{r}{R}
$$

## (Mean) Reciprocal Rank
---
**Reciprocal Rank** is the multiplicative inverse of the rank of the first retrieved relevant document: 1 for first place, 1/2 for second place, 1/3 for third place, and so on. When averaged over many queries, it is usually called **Mean Reciprocal Rank** (MRR).

$$
Reciprocal Rank = \frac{1}{rank}
$$

where,

- $rank$ is the position of the first retrieved relevant document.

## (Mean) Average Precision
---
**Average Precision** is the average of the Precision scores computed after each relevant document is retrieved. When averaged over many queries, it is usually called **Mean Average Precision** (MAP).

$$
\operatorname{Average Precision} = \frac{\sum_r \operatorname{Precision}@r}{R}
$$

where,

- $r$ is the position of a relevant document;
- $R$ is the total number of relevant documents.

## NDCG
---
Compute **Normalized Discounted Cumulative Gain** (NDCG) as proposed by [Järvelin et al.](http://doi.acm.org/10.1145/582415.582418).

<details>
    <summary>BibTeX</summary>
        ```bibtex
            @article{DBLP:journals/tois/JarvelinK02,
                author    = {Kalervo J{\"{a}}rvelin and
                            Jaana Kek{\"{a}}l{\"{a}}inen},
                title     = {Cumulated gain-based evaluation of {IR} techniques},
                journal   = {{ACM} Trans. Inf. Syst.},
                volume    = {20},
                number    = {4},
                pages     = {422--446},
                year      = {2002}
            }
        ```
</details>

$$
\operatorname{nDCG} = \frac{\operatorname{DCG}}{\operatorname{IDCG}}
$$

where,

- $\operatorname{DCG}$ is Discounted Cumulative Gain;
- $\operatorname{IDCG}$ is Ideal Discounted Cumulative Gain (max possibile DCG).

## NDCG Burges
---
Compute **Normalized Discounted Cumulative Gain** (NDCG) at k as proposed by [Burges et al.](https://doi.org/10.1145/1102351.1102363).

<details>
    <summary>BibTeX</summary>
        ```bibtex
            @inproceedings{DBLP:conf/icml/BurgesSRLDHH05,
                author    = {Christopher J. C. Burges and
                            Tal Shaked and
                            Erin Renshaw and
                            Ari Lazier and
                            Matt Deeds and
                            Nicole Hamilton and
                            Gregory N. Hullender},
                title     = {Learning to rank using gradient descent},
                booktitle = {{ICML}},
                series    = {{ACM} International Conference Proceeding Series},
                volume    = {119},
                pages     = {89--96},
                publisher = {{ACM}},
                year      = {2005}
            }
        ```
</details>

$$
\operatorname{nDCG} = \frac{\operatorname{DCG}}{\operatorname{IDCG}}
$$

where,

- $\operatorname{DCG}$ is Discounted Cumulative Gain;
- $\operatorname{IDCG}$ is Ideal Discounted Cumulative Gain (max possibile DCG).