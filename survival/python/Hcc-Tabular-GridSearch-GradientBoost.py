import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

feature_selected = True
seed = 66
test_ratio = 0.20
param_grid = {
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2],
    "max_depth": [8, 12, 16, 20, 24, 28, 32],
    "max_features": ["log2", "sqrt"],
    "criterion": ["friedman_mse", "mae"],
    "n_estimators": [50, 100, 200, 300, 400, 500]
}

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

clf_grid = GridSearchCV(GradientBoostingClassifier(random_state=seed), param_grid, verbose=1, n_jobs=10, cv=10)

clf_grid.fit(X_train, y_train)

print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)
print("Best Score:\n", clf_grid.best_score_)