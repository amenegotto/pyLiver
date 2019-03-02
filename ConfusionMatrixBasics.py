from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot

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

auc_score = roc_auc_score(y_true, y_pred)

print(auc_score)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()