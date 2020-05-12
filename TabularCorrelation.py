import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

np.set_printoptions(precision=3)
df = pd.read_csv('csv/clinical_data.csv')

X = df.drop(['Source', 'Patient', 'Hcc'], axis=1)
y = df['Hcc']

test = SelectKBest(score_func=chi2, k=20)
fit = test.fit(X, y)

cols = fit.get_support(indices=True)

X = X[X.columns[cols]]
#print("Selected Features: ")
#print(X.columns)

scores = test.scores_
feats = [ round(x,5) for x in scores]

for i in range(len(feats)):
    print(X.columns[i], ' = ', feats[i])
