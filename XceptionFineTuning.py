# PURPOSE:
# Xception fine tuning for hepatocarcinoma diagnosis through CTs images
# with image augmentation

from keras.layers import *
from keras.applications import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Summary import create_results_dir, get_base_name, plot_train_stats, write_summary_txt, copy_to_s3
from ExecutionAttributes import ExecutionAttribute
from TimeCallback import TimeCallback
from TrainingResume import save_execution_attributes
from keras.utils import plot_model
from Datasets import create_image_generator
import multiprocessing
from keras import backend as K

# fix seed for reproducible results (only works on CPU, not GPU)
# seed = 9
# np.random.seed(seed=seed)
# tf.set_random_seed(seed=seed)

# Summary Information
SUMMARY_PATH = "/mnt/data/results"
NETWORK_FORMAT = "Unimodal"
IMAGE_FORMAT = "2D"
SUMMARY_BASEPATH = create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)

# Execution Attributes
attr = ExecutionAttribute()
attr.architecture = 'Xception'

results_path = create_results_dir(SUMMARY_BASEPATH, 'fine-tuning', attr.architecture)
attr.summ_basename = get_base_name(results_path)
attr.s3_path = NETWORK_FORMAT + '/' + IMAGE_FORMAT
attr.path = '/mnt/data/image/2d/sem_pre_proc'
attr.set_dir_names()
attr.batch_size = 128  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
attr.epochs = 50

# hyper parameters for model
nb_classes = 2  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
attr.img_width, attr.img_height = 299, 299  # change based on the shape/structure of your images
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum

# how many times to execute the training/validation/test cycle
CYCLES = 5

for i in range(0, CYCLES):

    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    base_model = Xception(input_shape=(attr.img_width, attr.img_height, 3), weights='imagenet', include_top=False)

    # Top Model Block
    x = GlobalAveragePooling2D()(base_model.output)
    hidden1 = Dense(512, activation='relu')(x)
    drop = Dropout(0.30)(hidden1)
    predictions = Dense(nb_classes, activation='softmax')(drop)

    # add your top layer block to your base model
    attr.model = Model(base_model.input, predictions)
    print(attr.model.summary())

    # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
    # # uncomment when choosing based_model_last_block_layer
    # for i, layer in enumerate(attr.model.layers):
    #     print(i, layer.name)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = create_image_generator(True, True)

    test_datagen = create_image_generator(True, False)

    attr.train_generator = train_datagen.flow_from_directory(attr.train_data_dir,
                                                        target_size=(attr.img_width, attr.img_height),
                                                        batch_size=attr.batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical')

    # save and look at how the data augmentations look like
    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')

    attr.validation_generator = test_datagen.flow_from_directory(attr.validation_data_dir,
                                                                  target_size=(attr.img_width, attr.img_height),
                                                                  batch_size=attr.batch_size,
                                                                  shuffle=True,
                                                                  class_mode='categorical')

    attr.test_generator = test_datagen.flow_from_directory(
            attr.test_data_dir,
            target_size=(attr.img_height, attr.img_width),
            batch_size=1,
            class_mode='categorical',
            shuffle=False)

    callbacks = [
        ModelCheckpoint(attr.curr_basename + "-mid-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=10, verbose=0)
    ]

    attr.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    plot_model(attr.model, to_file=attr.summ_basename + '-architecture.png')

    # calculate steps based on number of images and batch size
    attr.calculate_steps()
    attr.increment_seq()

    # Train Simple CNN
    attr.model.fit_generator(attr.train_generator,
                        steps_per_epoch=attr.steps_train,
                        epochs=attr.epochs,
                        validation_data=attr.validation_generator,
                        validation_steps=attr.steps_valid,
                        use_multiprocessing=True,
                        workers=multiprocessing.cpu_count() - 1,
                        callbacks=callbacks)

    # verbose
    print("\nStarting to Fine Tune Model\n")

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
    attr.model.load_weights(attr.curr_basename + "-mid-ckweights.h5")

    # based_model_last_block_layer_number points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
    for layer in attr.model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in attr.model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    attr.model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    time_callback = TimeCallback()

    # save weights of best training epoch: monitor either val_loss or val_acc
    callbacks_list = [time_callback,
        ModelCheckpoint(attr.curr_basename + "-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=10, verbose=0)
    ]

    # Persist execution attributes for session resume
    save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

    # fine-tune the model
    history = attr.model.fit_generator(attr.train_generator,
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
    fnames = attr.test_generator.filenames

    # Get the ground truth from generator
    ground_truth = attr.test_generator.classes

    # Get the label to class mapping from the generator
    label2index = attr.test_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())

    # Get the predictions from the model using the generator
    predictions = attr.model.predict_generator(attr.test_generator, steps=attr.steps_test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    res = "No of errors = {}/{}".format(len(errors), attr.test_generator.samples)
    with open(attr.curr_basename + "-predicts.txt", "a") as f:
        f.write(res)
        print(res)
        f.close()

    # Reset test generator before summary predictions
    attr.test_generator.reset()

    write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback, callbacks_list[2].stopped_epoch)

    K.clear_session()

copy_to_s3(attr)
# os.system("sudo poweroff")
