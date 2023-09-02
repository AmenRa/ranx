# FAQ

## Is `ranx` suited for evaluating classification tasks?
No, it's not. `ranx` is meant for ranking tasks. Although some metrics are commonly used for evaluation of both tasks (e.g., `precision` and `recall`) the relevance scores stored in `runs` should not be confused with the predicted class labels of a classification task. Relevance scores are used by `ranx` to sort results before computing the metrics, regardless of their actual values.

## Are zero and negative scored results filtered out by `ranx`?
Zero and negative scored results are NOT filtered out by `ranx`.
Relevance scores are used only for sorting and there is no constraint on the values produce by a ranking models, although some of them only outputs positive values.
Therefore, if you think that zero and negative scored results should be filtered out, you should do it before passing the `runs` to `ranx`.