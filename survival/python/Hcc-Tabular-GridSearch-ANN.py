import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

feature_selected = True
seed = 66
test_ratio = 0.20
param_grid = {
    'batch_size': [4, 8, 16, 32],
    'max_iter': [100, 500, 1000, 3000],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
    'hidden_layer_sizes': [12, 24, 36, 48, 60]
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

clf_grid = GridSearchCV(MLPClassifier(random_state=seed), param_grid, verbose=0, n_jobs=1, cv=10)

clf_grid.fit(X_train, y_train)

print("Features Used: ")
print(X.columns)
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)
print("Best Score:\n", clf_grid.best_score_)
