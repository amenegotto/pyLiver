
# PURPOSE:
# Random forest regression for numeric missing data imputation (exams results)
# It should be used when there are only one missing column per row

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import model_selection

import matplotlib.pyplot as plt

column_name = 'Creatinine'
log_importance = False
performance_range = False
use_crossvalidation = True


def create_regressor(column_name, print_statistics = False, log_importance = False, performance_range = False, use_crossvalidation = True):

    df = pd.read_csv('csv/hcc-data-spline-best-features.csv')

    # here divides in labeles / target. In my case should do for each laboratory exam result
    X = df.iloc[:, df.columns != column_name].values
    y = df.iloc[:, df.columns.get_loc(column_name)].values

    # divides in test/validation with 90/10
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


        if print_statistics:
            y_pred = regressor.predict(X_test)
            print("Input=%s, Predicted=%s" % (X_test[0], y_pred[0]))
            print('R2 Score:', metrics.r2_score(y_test, y_pred))
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    else:
        cv = model_selection.KFold(n_splits=2)

        for train_index, test_index in cv.split(X):
            print('TRAIN:', train_index, 'TEST:', test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            regressor.set_params(n_estimators=62)
            regressor.fit(X_train, y_train)
        
        if print_statistics:
            y_pred = regressor.predict(X_test)
            print("Input=%s, Predicted=%s" % (X_test[0], y_pred[0]))
            print('R2 Score:', metrics.r2_score(y_test, y_pred))
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    if log_importance:
        feature_list = list(df.iloc[:, df.columns != column_name].columns)

        # Get numerical feature importances
        importances = list(regressor.feature_importances_)

        # Print out the feature and importances
        print(importances)

    return regressor


def populate_column(column, df):
    regressor = create_regressor(column) 
    
    for i, r in df.iterrows():
        if pd.isnull(r[column]) or pd.isna(r[column]):
            df.at[i, column] = regressor.predict(r.values())[0]
 

df = pd.read_csv('csv/hcc-missing-column.csv')
populate_column('AFP', df)
populate_column('Platelets', df)
populate_column('Prothrombin Time', df)
populate_column('Albumin', df)
populate_column('Total Bilirubin', df)
populate_column('Creatinine', df)

df.to_csv('csv/hcc-missing-column-filled-rf.csv')
