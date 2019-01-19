from sklearn.metrics import classification_report, confusion_matrix

# PURPOSE:
# experiments with scikit-learn confusion matrix to align labels X results

y_true = [1, 0, 1, 0, 1, 0, 1, 0]
y_pred = [1, 0, 0, 0, 1, 0, 1, 0]

mtx = confusion_matrix(y_true, y_pred, labels=[1, 0])
print('Confusion Matrix:')
print('TP   FP')
print('FN   TN')

print(mtx)


print(classification_report(y_true, y_pred, target_names=['caro', 'barato']))