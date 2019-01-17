# PURPOSE:
# unimodal DCNN for hepatocarcinoma computer-aided diagnosis

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# figure output
OUTPUT_FILENAME = '/home/amenegotto/dataset/2d/results/sem_pre_proc.png'

# dimensions of our images.
img_width, img_height = 96, 96

# network parameters
path='/home/amenegotto/dataset/2d/sem_pre_proc/'
#path='/mnt/images/com_pre_proc/'
train_data_dir = path + 'train'
validation_data_dir = path + 'valid'
test_data_dir = path + 'test'
epochs = 20
batch_size = 100

if K.image_data_format() == 'channels_first':
    input_s = (3, img_width, img_height)
else:
    input_s = (img_width, img_height, 3)

#define model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_s))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), input_shape=input_s))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(48, (3, 3), input_shape=input_s))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), input_shape=input_s))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3), input_shape=input_s))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), input_shape=input_s))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#compile model using accuracy as main metric, rmsprop (gradient descendent)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
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

history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=10)

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
plt.savefig(OUTPUT_FILENAME)

# test model
score = model.evaluate_generator(generator=test_generator, steps=20)

print('Test accuracy:', score[1])
