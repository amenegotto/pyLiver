# late fusion for car price classification 

import numpy as np # linear algebra

# keras libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate 
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from keras.utils import plot_model

# dimensions of our images.
img_width, img_height = 150, 150

# network parameters
train_data_dir = 'C:/Users/hp/Downloads/cars_train/trein'
validation_data_dir = 'C:/Users/hp/Downloads/cars_train/valid'
test_data_dir = 'C:/Users/hp/Downloads/cars_train/test'
nb_train_samples = 12
nb_validation_samples = 5
epochs = 50
batch_size = 5

if K.image_data_format() == 'channels_first':
    input_s = (3, img_width, img_height)
else:
    input_s = (img_width, img_height, 3)

#image input
img_input = Input(shape=input_s)
conv1 = Conv2D(32, kernel_size=4, activation='relu')(img_input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(32, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(64, kernel_size=4, activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
flatten_img = Flatten()(pool3)

#price input
price_input = Input(shape=(1,))

#merged input
merged = concatenate([flatten_img, price_input])
classifier = Dense(64, activation='relu', name='classifier_layer')(merged)
classifier = Dropout(0.5)(classifier)
out = Dense(1, activation='sigmoid', name='output_layer')(classifier)
model = Model(inputs = [img_input, price_input], outputs = [out])

#compile model using accuracy as main metric, rmsprop (gradient descendent)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#plot_model(model, to_file='multiple_inputs.png')


# this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator(
#        rotation_range=40,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
#test_datagen = ImageDataGenerator(rescale=1. / 255)

#train_generator = train_datagen.flow_from_directory(
#    train_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode='binary')

#validation_generator = test_datagen.flow_from_directory(
#    validation_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode='binary')

#test_generator = test_datagen.flow_from_directory(
#    test_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode='binary')

#history = model.fit_generator(
#    train_generator,
#    steps_per_epoch=20,
#    epochs=epochs,
#    validation_data=validation_generator,
#    validation_steps=5)

# plot history
#loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
#val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
#acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
#val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
#if len(loss_list) == 0:
#    print('Loss is missing in history')
       
## As loss always exists
#epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
## Loss
#import matplotlib.pyplot as plt

#plt.figure(1)
#for l in loss_list:
#    plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
#for l in val_loss_list:
#    plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
#plt.title('Loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
    
## Accuracy
#plt.figure(2)
#for l in acc_list:
#    plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
#for l in val_acc_list:    
#    plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

#plt.title('Accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.show()

# test model
#score = model.evaluate_generator(generator=test_generator, steps=20)

#print('Test accuracy:', score[1])

