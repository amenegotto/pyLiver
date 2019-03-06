# PURPOSE:
# test a structure for resume training after session is lost 

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Summary import plot_train_stats, create_results_dir, get_base_name, write_summary_txt, save_model
from TrainingResume import save_execution_attributes, read_attributes
from TimeCallback import TimeCallback
import tensorflow as tf
import numpy as np

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)


# Summary Information
# SUMMARY_PATH="/mnt/data/results"
# SUMMARY_PATH="c:/temp/results"
SUMMARY_PATH = "/tmp/results"
NETWORK_FORMAT = "Unimodal"
IMAGE_FORMAT = "2D"
SUMMARY_BASEPATH = create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)

# how many times to execute the training/validation/test
CYCLES = 1

#
# Execution Attributes
INITIAL_EPOCH = 1
attr = read_attributes('/tmp/results/Unimodal/2D/20190201-141755-execution-attributes.properties') 

if K.image_data_format() == 'channels_first':
    input_s = (1, attr.img_width, attr.img_height)
else:
    input_s = (attr.img_width, attr.img_height, 1)

for i in range(0, CYCLES):
    # define model
    attr.model = load_model(attr.summ_basename + '-ckweights.h5')

    time_callback = TimeCallback()

    callbacks = [time_callback, EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True),
                 ModelCheckpoint(attr.summ_basename + "-ckweights.h5", mode='min', verbose=1, monitor='val_loss', save_best_only=True)]

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        #    rotation_range=2,
         #   width_shift_range=0.2,
         #   height_shift_range=0.2,
            rescale=1./255,
         #   shear_range=0.2,
        #    zoom_range=0.1,
         #   horizontal_flip=True,
         #   fill_mode='nearest')
		)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(
            rescale=1. / 255
            )

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

    # Persist execution attributes for session resume
    save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

    # training time
    history = attr.model.fit_generator(
        attr.train_generator,
        steps_per_epoch=attr.steps_train,
        epochs=attr.epochs,
        validation_data=attr.validation_generator,
        validation_steps=attr.steps_valid,
        use_multiprocessing=False,
        callbacks=callbacks,
        initial_epoch=INITIAL_EPOCH)

    # plot loss and accuracy
    plot_train_stats(history, attr.curr_basename + '-training_loss.png', attr.curr_basename + '-training_accuracy.png')

    # make sure that the best weights are loaded (even if restore_best_weights is already true)
    attr.model.load_weights(filepath=attr.summ_basename + "-ckweights.h5")

    # save model with weights for later reuse
    save_model(attr)

    # create confusion matrix and report with accuracy, precision, recall, f-score
    write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback)
