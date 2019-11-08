import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

feature_selected = True
test_ratio = 0.20
seed = 66
pos = 0
param_max_depth = [3, 10]
param_min_samples_leaf = [1, 1]
param_min_samples_split = [2, 2]
param_criterion = ['gini', 'entropy']

data = pd.read_csv('../csv/hcc-data-complete-balanced.csv')

X = data.drop('Class', axis=1)
y = data['Class']

if feature_selected:
    pos = 1
    test = SelectKBest(score_func=chi2, k=20)
    fit = test.fit(X, y)

    cols = fit.get_support(indices=True)

    X = X[X.columns[cols]]
    print("Selected Features: ")
    print(X.columns)
else:
    pos = 0
    print("Used all features!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

clf = DecisionTreeClassifier(max_depth=param_max_depth[pos],
                             min_samples_leaf=param_min_samples_leaf[pos],
                             min_samples_split=param_min_samples_split[pos],
                             criterion=param_criterion[pos],
                             random_state=seed)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
clf_score = cross_validate(clf, X, y, cv=10, scoring=metrics, return_train_score=False, return_estimator=True)

print("==================================")
print("=      RANDOM FOREST             =")
print("==================================\n\n")
print("=== HOLD-OUT ===")
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n\n')
print('=== CROSS-VALIDATION: ====\n')
print("Avg Accuracy = ", np.mean(clf_score['test_accuracy']), "Std = ", np.std(clf_score['test_accuracy']))
print("Avg Recall = ", np.mean(clf_score['test_recall']), "Std = ", np.std(clf_score['test_recall']))
print("Avg Precision = ", np.mean(clf_score['test_precision']), "Std = ", np.std(clf_score['test_precision']))
print("Avg F1-Score = ", np.mean(clf_score['test_f1']), "Std = ", np.std(clf_score['test_f1']))
print("Avg ROC AUC = ", np.mean(clf_score['test_roc_auc']), "Std = ", np.std(clf_score['test_roc_auc']))
