# PURPOSE:
# Artificial Neural Network classification for hcc, aiming to experiment with different
# network architectures before fusion

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

def create_model():
    model = Sequential()
    model.add(Dense(6, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(column_name, model, plot_loss = False, plot_accuracy = False):

    df_train = pd.read_csv('csv/clinical_data.csv')

    df_train = df_train.drop('Source', axis=1)
    df_train = df_train.drop('Patient', axis=1)

    # here divides in labeles / target. In my case should do for each laboratory exam result
    X = df_train.iloc[:, df_train.columns != column_name].values
    y = df_train.iloc[:, df_train.columns.get_loc(column_name)].values

    # divides in test/validation with 90/10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    history = model.fit(X_train, y_train, epochs=100, batch_size=5,  verbose=0, validation_split=0.2)

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

    if plot_accuracy:
        print(history.history.keys())
        # "Loss"
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

model = create_model()
train_model('Hcc', model, False, True)
