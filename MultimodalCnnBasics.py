# PURPOSE:
# first experiments with a multimodal CNN architecture

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from ExecutionAttributes import ExecutionAttribute
from keras.utils import plot_model
from skimage import io as io
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support as score


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# Execution Attributes
attr = ExecutionAttribute()

# dimensions of our images.
attr.img_width, attr.img_height = 32, 32

# network parameters
attr.path = 'C:/Users/hp/Downloads/cars_train'
attr.epochs = 200
attr.batch_size = 8
attr.set_dir_names()

if K.image_data_format() == 'channels_first':
    input_s = (3, attr.img_width, attr.img_height)
else:
    input_s = (attr.img_width, attr.img_height, 3)


def load_data(filepath):
    files = pd.read_csv(filepath)

    images = []
    prices = []
    labels = []

    for row in files.itertuples():
        print(row.path + ' - ' + str(row.price))

        # img = io.imread(attr.path + '/' + row.path, as_grey=False)
        # img = img.reshape([attr.img_width, attr.img_height, 3])

        image = cv2.imread(attr.path + '/' + row.path)
        image = cv2.resize(image, (attr.img_width, attr.img_height))

        images.append(image)
        prices.append(row.price)
        if "barato" in row.path:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(images), np.array(prices), np.array(labels)


images_train, prices_train, labels_train = load_data(attr.path + '/trein.csv')
images_valid, prices_valid, labels_valid = load_data(attr.path + '/valid.csv')
images_test, prices_test, labels_test = load_data(attr.path + '/test.csv')

# define CNN for image
image_input = Input(shape=input_s)

x = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(image_input)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.25)(x)
x = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.25)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)

x = Flatten()(x)

attributes_input = Input(shape=(1,))

# as in this POC we have only one auxiliary variable, there's no need for another ANN, just concat in flatten before FC
concat = concatenate([x, attributes_input])

x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(concat)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.25)(x)
x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)

image_output = Dense(1, activation='sigmoid')(x)
image_model = Model(inputs=[image_input, attributes_input], outputs=image_output)


# define ANN for additional data
# attributes_input = Input(shape=(1,))
# y = Dense(32, activation='relu')(attributes_input)
# y = Dropout(0.25)(y)
# y = Dense(64, activation='relu')(y)
# y = Dropout(0.25)(y)
# y = Dense(32, activation='relu')(y)
# attributes_output = Dense(1, activation='sigmoid')(y)
# attributes_model = Model(inputs=attributes_input, outputs=attributes_output)

# late fusion both models
# combinedInput = concatenate([image_model.output, attributes_model.output])

# merged = Dense(2, activation='relu')(combinedInput)
# merged_output = Dense(1, activation='sigmoid')(merged)


# attr.model = Model(inputs=[image_input, attributes_input], outputs=merged_output)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
image_model.compile(optimizer=opt)

print(labels_train.shape())
print(attributes_input.shape())


# train the model
print("[INFO] training model...")
image_model.fit(
	[images_train, prices_train], labels_train,
	validation_data=([images_valid, prices_valid], labels_valid),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
Y_pred = image_model.predict([images_test, prices_test])
y_pred = np.argmax(Y_pred, axis=1)

print(Y_pred)
print(y_pred)

mtx = confusion_matrix(labels_test, y_pred)
print('Confusion Matrix:')
print(mtx)

print(classification_report(labels_test, y_pred))

cohen_score = cohen_kappa_score(labels_test, y_pred)
print("Kappa Score = " + str(cohen_score))

auc_score = roc_auc_score(labels_test, y_pred)
print("ROC AUC Score = " + str(auc_score))