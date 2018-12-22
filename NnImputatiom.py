# PURPOSE:
# Artificial Neural Network regression for numeric missing data imputation

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

COLUMN_NAME = 'Creatinine'

df = pd.read_csv('csv/hcc-data-spline-best-features.csv')

# here divides in labeles / target. In my case should do for each laboratory exam result
X = df.iloc[:, df.columns != COLUMN_NAME].values
y = df.iloc[:, df.columns.get_loc(COLUMN_NAME)].values


#scaler = MinMaxScaler()
#print(scaler.fit(X))
#print(scaler.fit(y))
#xscale=scaler.transform(X)
#yscale=scaler.transform(y)

# divides in test/validation with 90/10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=1000, batch_size=5,  verbose=1, validation_split=0.1)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

Xnew = np.array([[1, 67, 95, 99, 3.4, 2.1]])
ynew=model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))