# Report

A `Report` instance is automatically generated as the results of a comparison.  
A `Report` provides a convenient way of inspecting a comparison results and exporting those in LaTeX for your scientific publications.  
By changing the values of the parameters `rounding_digits` (int) and `show_percentages` (bool) you can control what is shown on printing and when generating LaTeX tables.

```python
from ranx import compare

# Compare different runs and perform statistical tests
report = compare(
    qrels=qrels,
    runs=[run_1, run_2, run_3, run_4, run_5],
    metrics=["map@100", "mrr@100", "ndcg@10"],
    max_p=0.01,  # P-value threshold
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
```python
print(report.to_latex())  # To get the LaTeX code
```