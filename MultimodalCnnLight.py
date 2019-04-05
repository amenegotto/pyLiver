# PURPOSE:
# multimodal DCNN for hepatocarcinoma computer-aided diagnosis
# with image augmentation , lightweight network architecture from scratch (DRAFT)

from keras.utils import plot_model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.models import Model
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
from Datasets import load_data, create_image_generator, multimodal_generator_two_inputs


# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# fix seed for reproducible results (only works on CPU, not GPU)
# seed = 9
# np.random.seed(seed=seed)
# tf.set_random_seed(seed=seed)

# Summary Information
SUMMARY_PATH = "/mnt/data/results"
# SUMMARY_PATH="c:/temp/results"
# SUMMARY_PATH="/tmp/results"
NETWORK_FORMAT = "Multimodal"
IMAGE_FORMAT = "2D"
SUMMARY_BASEPATH = create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)
INTERMEDIATE_FUSION = False
LATE_FUSION = True

# how many times to execute the training/validation/test cycle
CYCLES = 1

#
# Execution Attributes
attr = ExecutionAttribute()

# numpy_path = '/home/amenegotto/dataset/2d/numpy/sem_pre_proc_mini/'
numpy_path = '/mnt/data/image/2d/numpy/sem_pre_proc/'
# dimensions of our images.
attr.img_width, attr.img_height = 96, 96

# network parameters
# attr.path='C:/Users/hp/Downloads/cars_train'
# attr.path='/home/amenegotto/dataset/2d/sem_pre_proc_mini/
attr.csv_path = 'csv/clinical_data.csv'
# attr.path = '/mnt/data/image/2d/com_pre_proc/'
attr.summ_basename = get_base_name(SUMMARY_BASEPATH)
attr.epochs = 5
attr.batch_size = 32
attr.set_dir_names()

if K.image_data_format() == 'channels_first':
    input_image_s = (3, attr.img_width, attr.img_height)
else:
    input_image_s = (attr.img_width, attr.img_height, 3)

input_attributes_s = (20,)

images_train, fnames_train, attributes_train, labels_train, \
    images_valid, fnames_valid, attributes_valid, labels_valid, \
    images_test, fnames_test, attributes_test, labels_test = load_data(numpy_path)

attr.fnames_test = fnames_test
attr.labels_test = labels_test

for i in range(0, CYCLES):

    # define model
    
    # image input
    visible = Input(shape=input_image_s)
    conv1 = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(visible)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(act1)
    conv2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(drop1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(act2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(pool1)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(act3)
    conv4 = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(drop3)
    bn4 = BatchNormalization()(conv4)
    act5 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(act5)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop4)

    flat = Flatten()(pool2)

    if INTERMEDIATE_FUSION:
        attr.fusion = "Intermediate Fusion"

        attributes_input = Input(shape=input_attributes_s)
        concat = concatenate([flat, attributes_input])

        hidden1 = Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(concat)
        bn5 = BatchNormalization()(hidden1)
        act6 = Activation('relu')(bn5)
        drop5 = Dropout(0.40)(act6)
        hidden2 = Dense(1024, activation='relu', kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0.0005))(drop5)
        drop6 = Dropout(0.40)(hidden2)
        output = Dense(1, activation='sigmoid')(drop6)

    if LATE_FUSION:
        attr.fusion = "Late Fusion"

        hidden1 = Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(flat)
        bn5 = BatchNormalization()(hidden1)
        act6 = Activation('relu')(bn5)
        drop5 = Dropout(0.40)(act6)
        hidden2 = Dense(1024, activation='relu', kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0.0005))(drop5)
        drop6 = Dropout(0.40)(hidden2)
        output_img = Dense(1, activation='sigmoid')(drop6)

        attributes_input = Input(shape=input_attributes_s)
        hidden3 = Dense(32, kernel_initializer='he_normal')(attributes_input)
        bn6 = BatchNormalization()(hidden3)
        act7 = Activation('relu')(bn6)
        drop6 = Dropout(0.25)(act7)
        hidden4 = Dense(64, kernel_initializer='he_normal')(drop6)
        bn7 = BatchNormalization()(hidden4)
        act8 = Activation('relu')(bn7)
        drop7 = Dropout(0.40)(act7)
        output_attributes = Dense(1, activation='sigmoid')(drop7)

        concat = concatenate([output_img, output_attributes])
        hidden5 = Dense(8, kernel_initializer='he_normal', activation='relu')(concat)
        output = Dense(1, activation='sigmoid')(hidden5)

    attr.model = Model(inputs=[visible, attributes_input], outputs=output)

    plot_model(attr.model, to_file=attr.summ_basename + '-architecture.png')

    # compile model using accuracy as main metric, rmsprop (gradient descendent)
    attr.model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.000001),
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = create_image_generator(False, True)

    # this is the augmentation configuration we will use for testing:
    # nothing is done.
    test_datagen = create_image_generator(False, False)

    attr.train_generator = multimodal_generator_two_inputs(images_train, attributes_train, labels_train, train_datagen, attr.batch_size)
    attr.validation_generator = multimodal_generator_two_inputs(images_valid, attributes_valid, labels_valid, test_datagen, attr.batch_size)
    attr.test_generator = multimodal_generator_two_inputs(images_test, attributes_test, labels_test, test_datagen, 1)

    # calculate steps based on number of images and batch size
    attr.train_samples = len(images_train)
    attr.valid_samples = len(images_valid)
    attr.test_samples = len(images_test)

    attr.calculate_steps()

    attr.increment_seq()

    # Persist execution attributes for session resume
    save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

    time_callback = TimeCallback()

    callbacks = [time_callback, EarlyStopping(monitor='val_acc', patience=3, mode='max', restore_best_weights=True),
                 ModelCheckpoint(attr.curr_basename + "-ckweights.h5", mode='max', verbose=1, monitor='val_acc', save_best_only=True)]

   
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
    attr.model.load_weights(filepath=attr.curr_basename + "-ckweights.h5")

    # save model with weights for later reuse
    save_model(attr)

    # delete ckweights to save space - model file already has the best weights
    os.remove(attr.curr_basename + "-ckweights.h5")

    # create confusion matrix and report with accuracy, precision, recall, f-score
    write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback, callbacks[1].stopped_epoch)

    K.clear_session()

copy_to_s3(attr)
# os.system("sudo poweroff")
