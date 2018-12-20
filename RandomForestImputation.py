
# PURPOSE:
# Random forest regression for missing data imputation (DRAFT VERSION)

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import model_selection

import matplotlib.pyplot as plt

COLUMN_NAME = 'Creatinine'
log_importance = False
performance_range = False
use_crossvalidation = False

df = pd.read_csv('csv/hcc-data-spline-best-features.csv')

# here divides in labeles / target. In my case should do for each laboratory exam result
X = df.iloc[:, df.columns != COLUMN_NAME].values
y = df.iloc[:, df.columns.get_loc(COLUMN_NAME)].values


# divides in test/validation with 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# the n_estimators value should be tested. Lower RMSE is better...
regressor = RandomForestRegressor(random_state=42)

if not use_crossvalidation:
    
    if performance_range:
        estimators = np.arange(10, 400, 10)
        scores = []
        for n in estimators:
            regressor.set_params(n_estimators=n)
            regressor.fit(X_train, y_train)
            scores.append(regressor.score(X_test, y_test))
        plt.title("Effect of n_estimators")
        plt.xlabel("n_estimator")
        plt.ylabel("score")
        plt.plot(estimators, scores)
        plt.show(block=True)
    else:
        regressor.set_params(n_estimators=62)
        regressor.fit(X_train, y_train) 

    y_pred = regressor.predict(X_test)

    print("Input=%s, Predicted=%s" % (X_test[0], y_pred[0]))
    print('R2 Score:', metrics.r2_score(y_test, y_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
else:
    cv = model_selection.KFold(n_splits=10)

    for train_index, test_index in cv.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.set_params(n_estimators=290)
        regressor.fit(X_train, y_train) 
        y_pred = regressor.predict(X_test)
        print("Input=%s, Predicted=%s" % (X_test[0], y_pred[0]))
        print('R2 Score:', metrics.r2_score(y_test, y_pred))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


if log_importance:
    feature_list = list(df.iloc[:, df.columns != COLUMN_NAME].columns)

    # Get numerical feature importances
    importances = list(regressor.feature_importances_)

    # Print out the feature and importances
    print(importances)
