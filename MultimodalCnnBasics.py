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
from TimeCallback import TimeCallback
from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# Execution Attributes
attr = ExecutionAttribute()

# dimensions of our images.
attr.img_width, attr.img_height = 150, 150

# network parameters
# attr.path='C:/Users/hp/Downloads/cars_train'
# attr.path='/home/amenegotto/dataset/2d/com_pre_proc/'
attr.path = '/mnt/data/image/2d/com_pre_proc'
attr.epochs = 200
attr.batch_size = 256
attr.set_dir_names()

if K.image_data_format() == 'channels_first':
    input_s = (1, attr.img_width, attr.img_height)
else:
    input_s = (attr.img_width, attr.img_height, 1)

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

x = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.25)(x)
x = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.25)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)

x = Flatten()(x)
x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.25)(x)
x = Dense(1024, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)

image_output = Dense(1, activation='sigmoid')(x)
image_model = Model(inputs=image_input, outputs=image_output)


# define ANN for additional data
attributes_input = Input(shape=(5,))
y = Dense(32, activation='relu')(attributes_input)
x = Dropout(0.25)(x)
y = Dense(64, activation='relu')(y)
x = Dropout(0.25)(x)
y = Dense(32, activation='relu')(y)
attributes_output = Dense(1, activation='sigmoid')(y)
attributes_model = Model(inputs=attributes_input, outputs=attributes_output)

# late fusion both models
combinedInput = concatenate([image_model.output, attributes_model.output])

merged = Dense(6, activation='relu')(combinedInput)
merged_output = Dense(1, activation='sigmoid')(merged)


attr.model = Model(inputs=[image_input, attributes_input], outputs=merged_output)


print(attr.model.summary())

plot_model(attr.model, to_file='c:/temp/multimodal.png', show_shapes=True, show_layer_names=True)

# compile model using accuracy as main metric, rmsprop (gradient descendent)
#attr.model.compile(loss='binary_crossentropy',
#              optimizer=Adam(lr=0.00001),
#              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator(
#        rotation_range=2,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.1,
#        horizontal_flip=True,
#        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
#test_datagen = ImageDataGenerator(
#        rescale=1. / 255
#        )

#attr.train_generator = train_datagen.flow_from_directory(
#    attr.train_data_dir,
#    target_size=(attr.img_width, attr.img_height),
#    batch_size=attr.batch_size,
#    shuffle=True,
#    color_mode='grayscale',
#    class_mode='binary')

#attr.validation_generator = test_datagen.flow_from_directory(
#    attr.validation_data_dir,
#    target_size=(attr.img_width, attr.img_height),
#    batch_size=attr.batch_size,
#    shuffle=True,
#    color_mode='grayscale',
#    class_mode='binary')

#attr.test_generator = test_datagen.flow_from_directory(
#    attr.test_data_dir,
#    target_size=(attr.img_width, attr.img_height),
##    batch_size=1,
#    shuffle=False,
#    color_mode='grayscale',
#    class_mode='binary')

# calculate steps based on number of images and batch size
#attr.calculate_steps()

# training time
#history = attr.model.fit_generator(
#    attr.train_generator,
#    steps_per_epoch=attr.steps_train,
#    epochs=attr.epochs,
#    validation_data=attr.validation_generator,
#    validation_steps=attr.steps_valid,
#    use_multiprocessing=False)