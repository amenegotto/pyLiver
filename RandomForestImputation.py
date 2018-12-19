
# PURPOSE:
# Random forest regression for missing data imputation (DRAFT VERSION)

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

df = pd.read_csv('csv/hcc-data-filled.csv')

# here divides in labeles / target. In my case should do for each laboratory exam result
#X = dataset.iloc[:, 0:4].values
#y = dataset.iloc[:, 4].values


# divides in test/validation with 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# the n_estimators value should be tested. Lower RMSE is better...
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# create X_test array based on TCGA clinical data for data imputation
# with one line and n columns
predicted = rf.predict(X_test)
print("Input=%s, Predicted=%s" % (X_test[0], predicted[0]))
