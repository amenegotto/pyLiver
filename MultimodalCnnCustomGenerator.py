# PURPOSE:
# multimodal DCNN for hepatocarcinoma computer-aided diagnosis
# with augmentation and lightweight network architecture from scratch.
# Images, clinical attributes and labels are read from disk in batches
# using a custom thread-safe generator.

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
from MultimodalGenerator import MultimodalGenerator
import multiprocessing

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# when running on p3.2xlarge
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

# fix seed for reproducible results (only works on CPU, not GPU)
# seed = 9
# np.random.seed(seed=seed)
# tf.set_random_seed(seed=seed)

# Summary Information
IMG_TYPE = "com_pre_proc/"
SUMMARY_PATH = "/mnt/data/results"
# SUMMARY_PATH="c:/temp/results"
# SUMMARY_PATH="/tmp/results"
NETWORK_FORMAT = "Multimodal"
IMAGE_FORMAT = "2D"
SUMMARY_BASEPATH = create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)
INTERMEDIATE_FUSION = True
LATE_FUSION = False

# how many times to execute the training/validation/test cycle
CYCLES = 1

# Execution Attributes
attr = ExecutionAttribute()

# dimensions of our images.
attr.img_width, attr.img_height = 96, 96

# network parameters
attr.csv_path = 'csv/clinical_data.csv'
attr.numpy_path = '/mnt/data/image/2d/numpy/' + IMG_TYPE
# attr.numpy_path = '/home/amenegotto/dataset/2d/numpy/' + IMG_TYPE
attr.path = '/mnt/data/image/2d/' + IMG_TYPE
attr.summ_basename = get_base_name(SUMMARY_BASEPATH)
attr.epochs = 20
attr.batch_size = 128
attr.set_dir_names()

if K.image_data_format() == 'channels_first':
    input_image_s = (3, attr.img_width, attr.img_height)
else:
    input_image_s = (attr.img_width, attr.img_height, 3)

input_attributes_s = (20,)

for i in range(0, CYCLES):

    # define model

    # image input
    visible = Input(shape=input_image_s)
    conv1 = Conv2D(128, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(visible)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(act1)
    conv2 = Conv2D(128, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(drop1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(act2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop2)
    flat = Flatten()(pool1)

    if INTERMEDIATE_FUSION:
        attr.fusion = "Intermediate Fusion"

        attributes_input = Input(shape=input_attributes_s)
        concat = concatenate([flat, attributes_input])

        hidden1 = Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(concat)
        act3 = Activation('relu')(hidden1)
        drop3 = Dropout(0.40)(act3)
        hidden2 = Dense(256, activation='relu', kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0.0005))(
            drop3)
        drop4 = Dropout(0.40)(hidden2)
        output = Dense(1, activation='sigmoid')(drop4)

    if LATE_FUSION:
        attr.fusion = "Late Fusion"
        hidden1 = Dense(256, kernel_regularizer=regularizers.l2(0.0005))(flat)
        act3 = Activation('relu')(hidden1)
        drop3 = Dropout(0.40)(act3)
        hidden2 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(
            drop3)
        drop4 = Dropout(0.40)(hidden2)
        output_img = Dense(1, activation='sigmoid')(drop4)

        attributes_input = Input(shape=input_attributes_s)
        hidden3 = Dense(32, activation='relu')(attributes_input)
        drop6 = Dropout(0.2)(hidden3)
        hidden4 = Dense(16, activation='relu')(drop6)
        drop7 = Dropout(0.2)(hidden4)
        output_attributes = Dense(1, activation='sigmoid')(drop7)

        concat = concatenate([output_img, output_attributes])
        hidden5 = Dense(4, activation='relu')(concat)
        output = Dense(1, activation='sigmoid')(hidden5)

    attr.model = Model(inputs=[visible, attributes_input], outputs=output)

    plot_model(attr.model, to_file=attr.summ_basename + '-architecture.png')

    # compile model using accuracy as main metric, rmsprop (gradient descendent)
    attr.model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.000001),
                  metrics=['accuracy'])

    attr.train_generator = MultimodalGenerator(
            npy_path = attr.numpy_path + '/train.npy', 
            batch_size = attr.batch_size, 
            height = attr.img_height, 
            width = attr.img_width, 
            channels = 3, 
            classes = 2, 
            should_shuffle = True,
            is_categorical = False, 
            is_debug = False, 
            width_shift = 0.2, 
            height_shift = 0.2, 
            rotation_angle = 15, 
            shear_factor = 10, 
            zoom_factor = 0.2)

    attr.validation_generator = MultimodalGenerator(
            npy_path = attr.numpy_path + '/valid.npy', 
            batch_size = attr.batch_size, 
            height = attr.img_height, 
            width = attr.img_width, 
            channels = 3, 
            classes = 2, 
            should_shuffle = True,
            is_categorical = False, 
            is_debug = False, 
            width_shift = 0.2, 
            height_shift = 0.2, 
            rotation_angle = 15, 
            shear_factor = 10, 
            zoom_factor = 0.2)

    attr.test_generator = MultimodalGenerator(
            npy_path = attr.numpy_path + '/test.npy', 
            batch_size = 1, 
            height = attr.img_height, 
            width = attr.img_width, 
            channels = 3, 
            classes = 2, 
            should_shuffle = False,
            is_categorical = False, 
            is_debug = False)

    print("[INFO] Calculating samples and steps...")
    attr.calculate_samples_len()

    attr.calculate_steps()

    attr.increment_seq()

    # Persist execution attributes for session resume
    save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

    time_callback = TimeCallback()

    callbacks = [time_callback, EarlyStopping(monitor='val_acc', patience=5, mode='max', restore_best_weights=True),
                 ModelCheckpoint(attr.curr_basename + "-ckweights.h5", mode='max', verbose=1, monitor='val_acc', save_best_only=True)]

   
    # training time
    history = attr.model.fit_generator(
        attr.train_generator,
        steps_per_epoch=attr.steps_train,
        epochs=attr.epochs,
        validation_data=attr.validation_generator,
        validation_steps=attr.steps_valid,
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count() - 1,
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

# copy_to_s3(attr)
# os.system("sudo poweroff")
