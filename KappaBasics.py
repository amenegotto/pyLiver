from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

# PURPOSE:
# experiments with scikit-learn kappa score

y_true = [1, 0, 1, 0, 1, 0, 1, 0]
y_pred = [1, 0, 0, 0, 1, 0, 1, 0]

mtx = confusion_matrix(y_true, y_pred, labels=[1, 0])
print('Confusion Matrix:')
print('TP   FP')
print('FN   TN')

print(mtx)


print(classification_report(y_true, y_pred, target_names=['caro', 'barato']))

cohen_score = cohen_kappa_score(y_true, y_pred)
print(cohen_score)