import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# dimensions of our images.

img_width, img_height = 64, 64

path = 'C:/Users/hp/Downloads/cars_train'
# path='/home/amenegotto/Downloads/'
train_data_dir = path + '/trein'
validation_data_dir = path + '/valid'
test_data_dir = path + '/test'
epochs = 5
batch_size = 10

# Image Generator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=1,
                                                        class_mode='binary')

if K.image_data_format() == 'channels_first':
    input_s = (3, img_width, img_height)
else:
    input_s = (img_width, img_height, 3)

# define model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_s))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), input_shape=input_s))
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
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

# Train
model.fit_generator(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID)

score = model.evaluate_generator(generator=test_generator)

print(score)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


# Confusion Matrix and Classification Report
test_generator.reset()
Y_pred=model.predict_generator(test_generator,verbose=1)

y_pred = np.argmax(Y_pred, axis=1)

print(Y_pred)
print(y_pred)
print(test_generator.classes)

# print('Confusion Matrix')
mtx = confusion_matrix(test_generator.classes, y_pred)
print(mtx)

plt.imshow(mtx, cmap='binary', interpolation='None')
plt.show()

# print('Classification Report')
target_names = ['barato', 'caro']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
