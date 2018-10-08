# basic image classification for car price problem 

import numpy as np # linear algebra

# keras libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import Callback

# dimensions of our images.
img_width, img_height = 150, 150

# network parameters
path='c:/users/hp/Downloads/cars_train/'
train_data_dir = path + 'trein'
validation_data_dir = path + 'valid'
test_data_dir = path + 'test'
nb_train_samples = 25
nb_validation_samples = 5
epochs = 10
batch_size = 15

if K.image_data_format() == 'channels_first':
    input_s = (3, img_width, img_height)
else:
    input_s = (img_width, img_height, 3)

#define model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_s))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#compile model using accuracy as main metric, rmsprop (gradient descendent)
model.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.4,
        height_shift_range=0.4,
        rescale=1./255,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='/tmp/log/', histogram_freq=0,
                          write_graph=True, write_images=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=25,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=5,
    callbacks=[tensorboard])

# plot history
loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
if len(loss_list) == 0:
    print('Loss is missing in history')
       
## As loss always exists
epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
## Loss
import matplotlib.pyplot as plt

plt.figure(1)
for l in loss_list:
    plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
for l in val_loss_list:
    plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
    
## Accuracy
plt.figure(2)
for l in acc_list:
    plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
for l in val_acc_list:    
    plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# test model
score = model.evaluate_generator(generator=test_generator, steps=20)

print('Test accuracy:', score[1])

