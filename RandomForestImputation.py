
# PURPOSE:
# Random forest regression for missing data imputation (DRAFT VERSION)

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import matplotlib.pyplot as plt

df = pd.read_csv('csv/hcc-data-best-features.csv')

# here divides in labeles / target. In my case should do for each laboratory exam result
X = df.iloc[:, df.columns != 'Alpha-Fetoprotein'].values
y = df.iloc[:, 2].values


# divides in test/validation with 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)


# the n_estimators value should be tested. Lower RMSE is better...
regressor = RandomForestRegressor(random_state=42)


estimators = np.arange(10, 700, 10)
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

y_pred = regressor.predict(X_test)
print("Input=%s, Predicted=%s" % (X_test[0], y_pred[0]))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


feature_list = list(df.iloc[:, df.columns != 'Alpha-Fetoprotein'].columns)

# Get numerical feature importances
importances = list(regressor.feature_importances_)

# Print out the feature and importances
print(importances)
