from metrics_eval import ndcg
import numpy as np

y_true = np.array([[[12, 0.5], [25, 0.3]], [[11, 0.4], [2, 0.6]]])
y_pred = np.array([[12, 234, 25, 36, 32, 35], [12, 11, 25, 36, 2, 35]])
k = 5
print(ndcg(np.array(y_true), np.array(y_pred), k))
