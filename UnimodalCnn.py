# PURPOSE:
# unimodal DCNN for hepatocarcinoma computer-aided diagnosis (DRAFT)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from ExecutionAttributes import ExecutionAttribute
from Summary import plot_train_stats, create_results_dir, get_base_name, write_summary_txt

# Summary Information
SUMMARY_PATH="c:/temp/results"
# SUMMARY_PATH="/tmp/results"
NETWORK_FORMAT="Unimodal"
IMAGE_FORMAT="2D"
SUMMARY_BASEPATH=create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)

# how many times to execute the training/validation/test
CYCLES = 5

#
# Execution Attributes
attr = ExecutionAttribute()

# dimensions of our images.
attr.img_width, attr.img_height = 64, 64

# network parameters
attr.path='C:/Users/hp/Downloads/cars_train'
# attr.path='/home/amenegotto/dataset/2d/sem_pre_proc/'
# attr.path='/mnt/data/image/2d/sem_pre_proc/'
attr.summ_basename=get_base_name(SUMMARY_BASEPATH)
attr.epochs = 10
attr.batch_size = 5
attr.set_dir_names()

if K.image_data_format() == 'channels_first':
    input_s = (3, attr.img_width, attr.img_height)
else:
    input_s = (attr.img_width, attr.img_height, 3)

for i in range(0, CYCLES):
    # define model
    attr.model = Sequential()
    #attr.model.add(Conv2D(64, (3, 3), input_shape=input_s))
    #attr.model.add(Activation('relu'))
    #attr.model.add(Conv2D(64, (3, 3), input_shape=input_s))
    #attr.model.add(Activation('relu'))
    #attr.model.add(MaxPooling2D(pool_size=(3, 3)))

    #attr.model.add(Conv2D(48, (3, 3), input_shape=input_s))
    #attr.model.add(Activation('relu'))
    #attr.model.add(Conv2D(48, (3, 3), input_shape=input_s))
    #attr.model.add(Activation('relu'))
    #attr.model.add(MaxPooling2D(pool_size=(3, 3)))

    attr.model.add(Conv2D(32, (3, 3), input_shape=input_s))
    attr.model.add(Activation('relu'))
    attr.model.add(Conv2D(32, (3, 3), input_shape=input_s))
    attr.model.add(Activation('relu'))
    attr.model.add(MaxPooling2D(pool_size=(3, 3)))

    attr.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    attr.model.add(Dense(32))
    attr.model.add(Activation('relu'))
    attr.model.add(Dropout(0.1))
    attr.model.add(Dense(1))
    attr.model.add(Activation('sigmoid'))

    # compile model using accuracy as main metric, rmsprop (gradient descendent)
    attr.model.compile(loss='binary_crossentropy',
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

    attr.train_generator = train_datagen.flow_from_directory(
        attr.train_data_dir,
        target_size=(attr.img_width, attr.img_height),
        batch_size=attr.batch_size,
        class_mode='binary')

    attr.validation_generator = test_datagen.flow_from_directory(
        attr.validation_data_dir,
        target_size=(attr.img_width, attr.img_height),
        batch_size=attr.batch_size,
        class_mode='binary')

    attr.test_generator = test_datagen.flow_from_directory(
        attr.test_data_dir,
        target_size=(attr.img_width, attr.img_height),
        batch_size=attr.batch_size,
        class_mode='binary')

    # calculate steps based on number of images and batch size
    attr.calculate_steps()


    attr.increment_seq()
    history = attr.model.fit_generator(
        attr.train_generator,
        steps_per_epoch=attr.steps_train,
        epochs=attr.epochs,
        validation_data=attr.validation_generator,
        validation_steps=attr.steps_valid)

    plot_train_stats(history, attr.curr_basename + '-training_loss.png', attr.curr_basename + '-training_accuracy.png')

    write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['barato', 'caro'])
