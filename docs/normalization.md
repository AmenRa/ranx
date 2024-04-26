# Normalization

`ranx` provides several result lists normalization strategies to be used conjunctly with the fusion methods.
Normalization aims at transforming the scores of a result list into new values to make them comparable with those of other normalized result lists, which is mandatory for correctly applying many of the provided fusion methods.
The normalization strategy to apply before fusion can be defined through the `norm` parameter of the functions `fuse` and `optimize_fusion` (defaults to `min-max`).

| **Normalization Strategies**                   | **Alias**         |
|------------------------------------------------|-------------------|
| [Min-Max Norm][min-max-norm]                   | min-max           |
| [Max Norm][max-norm]                           | max               |
| [Sum Norm][sum-norm]                           | sum               |
| [ZMUV Norm][zmuv-norm]                         | zmuv              |
| [Rank Norm][rank-norm]                         | rank              |
| [Borda Norm][borda-norm]                       | borda             |


## Min-Max Norm
---
Min-Max Norm scales the scores (s) of a result list between 0 and 1, scaling to 0 the minimum score ($s_{min}$) and 1 the maximum score ($s_{max}$).

$$
\operatorname{MinMaxNorm(s)}=\frac{s - s_{min}}{s_{max} - s_{min}}
$$

Min-Max Norm accepts an optional boolean parameter `invert`, which, when set to true,
Min-Max Norm scales the scores (s) of a result list between 0 and 1,
setting the maximum score ($s_{max}$) to 0 and the minimum score ($s_{min}$) to 1.

$$
\operatorname{MinMaxNorm(s)}=\frac{s_{max} - s}{s_{max} - s_{min}}
$$

## Max Norm
---
Max Norm scales the scores (s) of a result list the maximum score ($s_{max}$) is scaled to 1.

$$
\operatorname{MaxNorm(s)}=\frac{s}{s_{max}}
$$

## Sum Norm
---
Sum Norm scales the minimum score ($s_{min}$) to 0 and the scores sum to 1. It is computed as follows:

$$
\operatorname{SumNorm(s)}=\frac{s - s_{min}}{\sum_s{s - s_{min}}}
$$

## ZMUV Norm
---
ZMUV Norm (zero-mean, unit-variance) scales the scores so that their mean ($s_{mean}$) becomes zero and their variance 1.

$$
\operatorname{ZMUVNorm(s)}=\frac{s - s_{mean}}{s_{std}}
$$

## Rank Norm
---
Rank Norm transforms the scores according to the position in the ranking of the results they are associated with.
In this case, the normalized scores are uniformly distributed.
The top-ranked result gets a score of 1, while the bottom-ranked result gets a score of $\frac{1}{|r|}$, where $|r|$ is the number of results in the ranked list.

$$
\operatorname{RankNorm(s_i)}=1-\frac{r_i - 1}{|r|}
$$

## Borda Norm
---
Borda Norm transforms the scores in a similar manner of how BordaFuse assign points to the results before fusing multiple runs.
Borda Norm is defined as follows:

$$
\operatorname{BordaNorm(s_i)}=
\begin{cases}
    1 - \frac{r_i - 1}{|candidates|} & \mathit{if}\ d \in r \\
    \frac{1}{2} - \frac{|r|-1}{2 \cdot |candidates|} & \mathit{otherwise}
\end{cases}
$$

Please, refer to [Renda et al.](https://dl.acm.org/doi/10.1145/952532.952698) for further details.