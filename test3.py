from sklearn.metrics import classification_report

y_true = [0, 0, 0, 1, 1, 0, 0, 2]
y_pred = [0, 0, 0, 0, 1, 1, 1, 2]

print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1', 'class 3']))