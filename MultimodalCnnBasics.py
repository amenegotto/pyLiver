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
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

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
    prices = files[['price']]
    labels = []

    for i, r in files.iterrows():
        print(r['path'] + ' - ' + str(r['price']))

        #image = load_img(attr.path + '/' + r['path'])
        #image.thumbnail((attr.img_width, attr.img_height))

        # image = io.imread()
        # image = image.reshape([attr.img_width, attr.img_height, 3])

        image = cv2.imread(attr.path + '/' + r['path'])
        image = cv2.resize(image, (attr.img_width, attr.img_height))
        image = img_to_array(image)
        images.append(image)

        if "barato" in r['path']:
            labels.append(0)
        else:
            labels.append(1)

    return (np.array(images, dtype="float") / 255.0), np.array(prices), np.array(labels)


images_train, prices_train, labels_train = load_data(attr.path + '/trein.csv')
images_valid, prices_valid, labels_valid = load_data(attr.path + '/valid.csv')
images_test, prices_test, labels_test = load_data(attr.path + '/test.csv')

print("[INFO] Training image size: {:.2f}MB".format(images_train.nbytes / (1024 * 1000.0)))
print("[INFO] Validation image size: {:.2f}MB".format(images_valid.nbytes / (1024 * 1000.0)))
print("[INFO] Testing image size: {:.2f}MB".format(images_test.nbytes / (1024 * 1000.0)))

# define CNN for image
visible = Input(shape=input_s)
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)

# as in this POC we have only one auxiliary variable, there's no need for another ANN, just concat in flatten before FC
attributes_input = Input(shape=(1,))
concat = concatenate([flat, attributes_input])

hidden1 = Dense(10, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=[visible, attributes_input], outputs=output)

# model.summary()
# plot_model(model, to_file='c:/temp/foo2.png')


opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss='binary_crossentropy', optimizer=opt,  metrics=['accuracy'])

# train the model
print("[INFO] training model...")
model.fit(
	[images_train, prices_train], labels_train,
	validation_data=([images_valid, prices_valid], labels_valid),
	epochs=20, batch_size=4)

# make predictions on the testing data
print("[INFO] predicting car prices...")
Y_pred = model.predict([images_test, prices_test])
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
