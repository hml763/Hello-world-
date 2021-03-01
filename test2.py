import numpy as np
from sklearn.metrics import precision_recall_curve

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0, 0, 0, 1])

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

print(precision)
print(recall)
print(thresholds)

print("\n\n\n\nTest5\nMAE : 4.46983")