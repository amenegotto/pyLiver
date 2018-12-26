# PURPOSE:
# Artificial Neural Network regression for numeric missing data imputation (exams results)
# It should be used when there are only one missing column per row

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

def create_model:
    model = Sequential()
    model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    
    return model

def train_model(column_name, model, plot_loss = False):

    df_train = pd.read_csv('csv/hcc-data-spline-best-features.csv')

    # here divides in labeles / target. In my case should do for each laboratory exam result
    X = df_train.iloc[:, df_train.columns != column_name].values
    y = df_train.iloc[:, df_train.columns.get_loc(column_name)].values

    # divides in test/validation with 90/10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    history = model.fit(X_train, y_train, epochs=1000, batch_size=5,  verbose=1, validation_split=0.1)

    if plot_loss:
        print(history.history.keys())
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

def populate_column(column, df):
    model = create_model
    train_model(column, model, False)
    for i, r in df.iterrows():
        if pd.isnull(r[column]) or pd.isna(r[column]):
            df.at[i, column] = model.predict(r.values())[0]
 

df = pd.read_csv('csv/hcc-missing-column.csv')
populate_column('AFP', df)
populate_column('Platelets', df)
populate_column('Prothrombin Time', df)
populate_column('Albumin', df)
populate_column('Total Bilirubin', df)
populate_column('Creatinine', df)

df.to_csv('csv/hcc-missing-column-filled-nn.csv')