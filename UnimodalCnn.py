# PURPOSE:
# unimodal DCNN for hepatocarcinoma computer-aided diagnosis (DRAFT)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from ExecutionAttributes import ExecutionAttribute
from TimeCallback import TimeCallback
from Summary import plot_train_stats, create_results_dir, get_base_name, write_summary_txt, save_model, copy_to_s3
from TrainingResume import save_execution_attributes
import os
import numpy as np
import tensorflow as tf

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# Summary Information
SUMMARY_PATH = "/mnt/data/results"
# SUMMARY_PATH="c:/temp/results"
# SUMMARY_PATH="/tmp/results"
NETWORK_FORMAT = "Unimodal"
IMAGE_FORMAT = "2D"
SUMMARY_BASEPATH = create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)

# how many times to execute the training/validation/test cycle
CYCLES = 1

#
# Execution Attributes
attr = ExecutionAttribute()

# dimensions of our images.
attr.img_width, attr.img_height = 150, 150

# network parameters
# attr.path='C:/Users/hp/Downloads/cars_train'
# attr.path='/home/amenegotto/dataset/2d/sem_pre_proc_mini/
attr.path = '/mnt/data/image/2d/com_pre_proc/'
attr.summ_basename = get_base_name(SUMMARY_BASEPATH)
attr.epochs = 200
attr.batch_size = 32
attr.set_dir_names()

if K.image_data_format() == 'channels_first':
    input_s = (1, attr.img_width, attr.img_height)
else:
    input_s = (attr.img_width, attr.img_height, 1)

for i in range(0, CYCLES):
    # define model
    attr.model = Sequential()
    attr.model.add(Conv2D(128, (3, 3), input_shape=input_s, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005)))
    attr.model.add(BatchNormalization())
    attr.model.add(Activation('relu'))
    attr.model.add(Dropout(0.25))
    attr.model.add(Conv2D(128, (3, 3), input_shape=input_s, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005)))
    attr.model.add(BatchNormalization())
    attr.model.add(Activation('relu'))
    attr.model.add(Dropout(0.25))
    attr.model.add(MaxPooling2D(pool_size=(3, 3)))

    attr.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    attr.model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005)))
    attr.model.add(Activation('relu'))
    attr.model.add(Dropout(0.25))
    attr.model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005)))
    attr.model.add(Dense(1))
    attr.model.add(Activation('sigmoid'))

    # compile model using accuracy as main metric, rmsprop (gradient descendent)
    attr.model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.000001),
                  metrics=['accuracy'])

    time_callback = TimeCallback()

    callbacks = [time_callback, EarlyStopping(monitor='val_acc', patience=10, mode='max', restore_best_weights=True),
                 ModelCheckpoint(attr.summ_basename + "-ckweights.h5", mode='max', verbose=1, monitor='val_acc', save_best_only=True)]

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
         #   rotation_range=2,
         #   width_shift_range=0.2,
         #   height_shift_range=0.2,
            rescale=1./255,
         #   shear_range=0.2,
         #   zoom_range=0.1,
         #   horizontal_flip=True,
         #   fill_mode='nearest')
		)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    attr.train_generator = train_datagen.flow_from_directory(
        attr.train_data_dir,
        target_size=(attr.img_width, attr.img_height),
        batch_size=attr.batch_size,
        shuffle=True,
        color_mode='grayscale',
        class_mode='binary')

    attr.validation_generator = test_datagen.flow_from_directory(
        attr.validation_data_dir,
        target_size=(attr.img_width, attr.img_height),
        batch_size=attr.batch_size,
        shuffle=True,
        color_mode='grayscale',
        class_mode='binary')

    attr.test_generator = test_datagen.flow_from_directory(
        attr.test_data_dir,
        target_size=(attr.img_width, attr.img_height),
        batch_size=1,
        shuffle=False,
        color_mode='grayscale',
        class_mode='binary')

    # calculate steps based on number of images and batch size
    attr.calculate_steps()

    attr.increment_seq()

    # Persist execution attributes for session resume
    save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

    # training time
    history = attr.model.fit_generator(
        attr.train_generator,
        steps_per_epoch=attr.steps_train,
        epochs=attr.epochs,
        validation_data=attr.validation_generator,
        validation_steps=attr.steps_valid,
        use_multiprocessing=True,
        callbacks=callbacks)

    # plot loss and accuracy
    plot_train_stats(history, attr.curr_basename + '-training_loss.png', attr.curr_basename + '-training_accuracy.png')

    # make sure that the best weights are loaded (even if restore_best_weights is already true)
    attr.model.load_weights(filepath=attr.summ_basename + "-ckweights.h5")

    # save model with weights for later reuse
    save_model(attr)

    # delete ckweights to save space - model file already has the best weights
    os.remove(attr.summ_basename + "-ckweights.h5")

    # create confusion matrix and report with accuracy, precision, recall, f-score
    write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback)

    K.clear_session()

copy_to_s3(attr)
# os.system("sudo poweroff")
