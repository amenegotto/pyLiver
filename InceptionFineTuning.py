# PURPOSE:
# InceptionV3 fine tuning for hepatocarcinoma diagnosis through CTs images
# with image augmentation

import os
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
import numpy as np
from Summary import create_results_dir, get_base_name, plot_train_stats, write_summary_txt, copy_to_s3
from ExecutionAttributes import ExecutionAttribute
from TimeCallback import TimeCallback
from TrainingResume import save_execution_attributes
from keras.utils import plot_model
from Datasets import create_image_generator


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
attr.architecture = 'InceptionV3'

results_path = create_results_dir(SUMMARY_BASEPATH, 'fine-tuning', attr.architecture)
attr.summ_basename = get_base_name(results_path)
attr.path = '/mnt/data/image/2d/com_pre_proc'
attr.set_dir_names()
attr.batch_size = 64  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
attr.epochs = 1

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# dimensions of our images.
# Inception input size
attr.img_width, attr.img_height = 299, 299

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
drop = Dropout(0.50)(x)

# and a logistic layer -- we have 2 classes
predictions = Dense(2, activation='softmax')(drop)

# this is the model we will train
attr.model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
attr.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], )

# prepare data augmentation configuration
train_datagen = create_image_generator(True, True)

test_datagen = create_image_generator(True, False)

attr.train_generator = train_datagen.flow_from_directory(
    attr.train_data_dir,
    target_size=(attr.img_height, attr.img_width),
    batch_size=attr.batch_size,
    shuffle=True,
    class_mode='categorical')

attr.validation_generator = test_datagen.flow_from_directory(
    attr.validation_data_dir,
    target_size=(attr.img_height, attr.img_width),
    batch_size=attr.batch_size,
    shuffle=True,
    class_mode='categorical')

attr.test_generator = test_datagen.flow_from_directory(
    attr.test_data_dir,
    target_size=(attr.img_height, attr.img_width),
    batch_size=1,
    shuffle=False,
    class_mode='categorical')

callbacks_top = [
    ModelCheckpoint(attr.summ_basename + "-mid-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, verbose=0)
]

# calculate steps based on number of images and batch size
attr.calculate_steps()

attr.increment_seq()

# Persist execution attributes for session resume
save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

attr.model.fit_generator(
    attr.train_generator,
    steps_per_epoch=attr.steps_train,
    epochs=attr.epochs,
    validation_data=attr.validation_generator,
    validation_steps=attr.steps_valid,
    use_multiprocessing=True,
    workers=10,
    callbacks=callbacks_top)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

attr.model.load_weights(attr.summ_basename + "-mid-ckweights.h5")

time_callback = TimeCallback()

#Save the model after every epoch.
callbacks_list = [time_callback,
    ModelCheckpoint(attr.summ_basename + "-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, verbose=0)
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
    workers=10,
    callbacks=callbacks_list)

# Save the model
attr.model.save(attr.summ_basename + '-weights.h5')

# Plot train stats
plot_train_stats(history, attr.summ_basename + '-training_loss.png', attr.summ_basename + '-training_accuracy.png')

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
with open(attr.summ_basename + "-predicts.txt", "a") as f:
    f.write(res)
    print(res)
    f.close()

write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback, callbacks_list[2].stopped_epoch)

# copy_to_s3(attr)
