# Statistical Tests

`ranx` provides two statistical tests that can be used when comparing different runs:  

- [Fisher's Randomization Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/fishrand.htm)
- [Two-sided Paired Student's t-Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/t_test.htm)

Please, refer to [Smucker et al.](https://dl.acm.org/doi/10.1145/1321440.1321528) for additional information on statistical tests for Information Retrieval.

To use the [Fisher's Randomization Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/fishrand.htm), pass `stat_test="fisher"` to [compare](https://amenra.github.io/ranx/compare).

To use the [Two-sided Paired Student's t-Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/t_test.htm), pass `stat_test="student"` to [compare](https://amenra.github.io/ranx/compare).