import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

feature_selected = True
seed = 66
test_ratio = 0.20
param_grid = {
      'base_estimator__criterion': ['gini', 'entropy'],
       'base_estimator__max_depth': [3, 10, 15, 20, 24, 28],
       'base_estimator__max_features': [4, 12, 16, 20],
       'bootstrap': [True, False],
       'bootstrap_features': [True, False],
       'n_estimators': [50, 100, 200, 300],
       'max_samples': [0.6, 0.8, 1.0],
}

base_cls = DecisionTreeClassifier()

data = pd.read_csv('../csv/hcc-data-complete-balanced.csv')

X = data.drop('Class', axis=1)
y = data['Class']

if feature_selected:
    test = SelectKBest(score_func=chi2, k=20)
    fit = test.fit(X, y)

    cols = fit.get_support(indices=True)

    X = X[X.columns[cols]]
    print("Selected Features: ")
    print(X.columns)
else:
    print("Used all features!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

clf_grid = GridSearchCV(BaggingClassifier(base_estimator=base_cls, random_state=seed), param_grid, verbose=1, n_jobs=6, cv=10)

clf_grid.fit(X_train, y_train)

print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)
print("Best Score:\n", clf_grid.best_score_)
