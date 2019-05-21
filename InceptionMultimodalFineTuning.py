# PURPOSE:
# InceptionV3 fine tuning for hepatocarcinoma diagnosis through CTs images
# with image augmentation and multimodal inputs

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, concatenate
from keras.layers import Dropout, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
import numpy as np
from Summary import create_results_dir, get_base_name, plot_train_stats, write_summary_txt, copy_to_s3
from ExecutionAttributes import ExecutionAttribute
from TimeCallback import TimeCallback
from TrainingResume import save_execution_attributes
from keras.utils import plot_model
from MultimodalGenerator import MultimodalGenerator
import multiprocessing
from keras import backend as K

# fix seed for reproducible results (only works on CPU, not GPU)
# seed = 9
# np.random.seed(seed=seed)
# tf.set_random_seed(seed=seed)

# Summary Information
IMG_TYPE = "sem_pre_proc/"
SUMMARY_PATH = "/mnt/data/results"
# SUMMARY_PATH="c:/temp/results"
# SUMMARY_PATH="/tmp/results"
NETWORK_FORMAT = "Multimodal"
IMAGE_FORMAT = "2D"
SUMMARY_BASEPATH = create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)
INTERMEDIATE_FUSION = True
LATE_FUSION = False

# Execution Attributes
attr = ExecutionAttribute()
attr.architecture = 'InceptionV3'
attr.numpy_path = '/mnt/data/image/2d/numpy/' + IMG_TYPE
attr.path = '/mnt/data/image/2d/' + IMG_TYPE

results_path = create_results_dir(SUMMARY_BASEPATH, 'fine-tuning', attr.architecture)
attr.summ_basename = get_base_name(results_path)
attr.s3_path = NETWORK_FORMAT + '/' + IMAGE_FORMAT
attr.set_dir_names()
attr.batch_size = 128  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
attr.epochs = 50

# how many times to execute the training/validation/test cycle
CYCLES = 5

for i in range(0, CYCLES):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # dimensions of our images.
    # Inception input size
    attr.img_width, attr.img_height = 299, 299

    input_attributes_s = (20,)

    # Top Model Block
    glob1 = GlobalAveragePooling2D()(base_model.output)
    hidout = Dense(1024, activation='relu')(glob1)
    drop = Dropout(0.20)(hidout)

    if INTERMEDIATE_FUSION:
        attr.fusion = "Intermediate Fusion"

        attributes_input = Input(shape=input_attributes_s)
        concat = concatenate([drop, attributes_input])
        output = Dense(2, activation='softmax')(concat)

        attr.model = Model(inputs=[base_model.input, attributes_input], outputs=output)

    if LATE_FUSION:
        attr.fusion = "Late Fusion"
        output_img = Dense(2, activation='softmax')(drop)

        model_img = Model(inputs=base_model.input, outputs=output_img)

        attributes_input = Input(shape=input_attributes_s)
        hidden3 = Dense(128, activation='relu')(attributes_input)
        drop6 = Dropout(0.20)(hidden3)
        hidden4 = Dense(64, activation='relu')(drop6)
        drop7 = Dropout(0.20)(hidden4)
        output_attributes = Dense(1, activation='sigmoid')(drop7)
        model_attr = Model(inputs=attributes_input, outputs=output_attributes)

        concat = concatenate([model_img.output, model_attr.output])

        hidden5 = Dense(8, activation='relu')(concat)
        output = Dense(2, activation='softmax')(hidden5)

        attr.model = Model(inputs=[model_img.input, model_attr.input], outputs=output)

    plot_model(attr.model, to_file=attr.summ_basename + '-architecture.png')

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    attr.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], )

    attr.train_generator = MultimodalGenerator(
                npy_path = attr.numpy_path + '/train-categorical.npy', 
                batch_size = attr.batch_size, 
                height = attr.img_height, 
                width = attr.img_width, 
                channels = 3, 
                classes = 2, 
                should_shuffle = True,
                is_categorical = True, 
                is_debug = False, 
                width_shift = 0.2, 
                height_shift = 0.2, 
                rotation_angle = 15, 
                shear_factor = 10, 
                zoom_factor = 0.2)

    attr.validation_generator = MultimodalGenerator(
                npy_path = attr.numpy_path + '/valid-categorical.npy', 
                batch_size = attr.batch_size, 
                height = attr.img_height, 
                width = attr.img_width, 
                channels = 3, 
                classes = 2, 
                should_shuffle = True,
                is_categorical = True, 
                is_debug = False, 
                width_shift = 0.2, 
                height_shift = 0.2, 
                rotation_angle = 15, 
                shear_factor = 10, 
                zoom_factor = 0.2)

    attr.test_generator = MultimodalGenerator(
                npy_path = attr.numpy_path + '/test-categorical.npy', 
                batch_size = 1, 
                height = attr.img_height, 
                width = attr.img_width, 
                channels = 3, 
                classes = 2, 
                should_shuffle = False,
                is_categorical = True, 
                is_debug = False)

    print("[INFO] Calculating samples and steps...")
    attr.calculate_samples_len()

    attr.calculate_steps()

    attr.increment_seq()

    callbacks_top = [
        ModelCheckpoint(attr.curr_basename + "-mid-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=10, verbose=0)
    ]

    # Persist execution attributes for session resume
    save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

    attr.model.fit_generator(
        attr.train_generator,
        steps_per_epoch=attr.steps_train,
        epochs=attr.epochs,
        validation_data=attr.validation_generator,
        validation_steps=attr.steps_valid,
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count() - 1,
        callbacks=callbacks_top)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    attr.model.load_weights(attr.curr_basename + "-mid-ckweights.h5")

    time_callback = TimeCallback()

    #Save the model after every epoch.
    callbacks_list = [time_callback,
        ModelCheckpoint(attr.curr_basename + "-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=10, verbose=0)
    ]

    # train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in attr.model.layers[:172]:
        layer.trainable = False
    for layer in attr.model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    attr.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    plot_model(attr.model, to_file=attr.summ_basename + '-architecture.png')

    history = attr.model.fit_generator(
        attr.train_generator,
        steps_per_epoch=attr.steps_train,
        epochs=attr.epochs,
        validation_data=attr.validation_generator,
        validation_steps=attr.steps_valid,
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count() - 1,
        callbacks=callbacks_list)

    # Save the model
    attr.model.save(attr.curr_basename + '-weights.h5')

    # Plot train stats
    plot_train_stats(history, attr.curr_basename + '-training_loss.png', attr.curr_basename + '-training_accuracy.png')

    # Reset test generator before raw predictions
    attr.test_generator.reset()

    # Get the filenames from the generator
    fnames = attr.fnames_test

    # Get the ground truth from generator
    ground_truth = attr.test_generator.get_labels() 

    # Get the predictions from the model using the generator
    predictions = attr.model.predict_generator(attr.test_generator, steps=attr.steps_test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    res = "No of errors = {}/{}".format(len(errors), len(attr.fnames_test))
    with open(attr.curr_basename + "-predicts.txt", "a") as f:
        f.write(res)
        print(res)
        f.close()

    # Reset test generator before summary predictions
    attr.test_generator.reset()

    write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback, callbacks_list[2].stopped_epoch)

    K.clear_session()

copy_to_s3(attr)
