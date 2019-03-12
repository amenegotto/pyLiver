# PURPOSE:
# VGG19 fine tuning for hepatocarcinoma diagnosis through CTs images

import os
import numpy as np
from keras.applications import VGG19
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Summary import create_results_dir, get_base_name, plot_train_stats, write_summary_txt, copy_to_s3
from ExecutionAttributes import ExecutionAttribute
from TimeCallback import TimeCallback
from TrainingResume import save_execution_attributes
import tensorflow as tf

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# Summary Information
SUMMARY_PATH = "/mnt/data/results"
NETWORK_FORMAT = "Unimodal"
IMAGE_FORMAT = "2D"
SUMMARY_BASEPATH = create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)

# Execution Attributes
attr = ExecutionAttribute()
attr.architecture = 'vgg19'

results_path = create_results_dir(SUMMARY_BASEPATH, 'fine-tuning', attr.architecture)
attr.summ_basename = get_base_name(results_path)
attr.path = '/mnt/data/image/2d/com_pre_proc'
attr.set_dir_names()
attr.batch_size = 256 
attr.epochs = 200

attr.img_width = 224
attr.img_height = 224

#Load the VGG model
vgg_conv = VGG19(weights='imagenet', include_top=False, input_shape=(attr.img_width, attr.img_height, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


# Create the model
attr.model = models.Sequential()
 
# Add the vgg convolutional base model
attr.model.add(vgg_conv)
 
# Add new layers
attr.model.add(layers.Flatten())
attr.model.add(layers.Dense(1024, activation='relu'))
attr.model.add(layers.Dropout(0.3))
attr.model.add(layers.Dense(2, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
attr.model.summary()

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.1,
    # zoom_range=0.1,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

attr.train_generator = train_datagen.flow_from_directory(
    attr.train_data_dir,
    target_size=(attr.img_height, attr.img_width),
    batch_size=attr.batch_size,
    class_mode='categorical')

# Create a generator for prediction
attr.validation_generator = test_datagen.flow_from_directory(
        attr.validation_data_dir,
        target_size=(attr.img_height, attr.img_width),
        batch_size=attr.batch_size,
        class_mode='categorical',
        shuffle=False)

attr.test_generator = test_datagen.flow_from_directory(
        attr.test_data_dir,
        target_size=(attr.img_height, attr.img_width),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

time_callback = TimeCallback()

callbacks = [time_callback, EarlyStopping(monitor='val_acc', patience=15, mode='max', restore_best_weights=True),
             ModelCheckpoint(attr.summ_basename + "-ckweights.h5", mode='max', verbose=1, monitor='val_acc', save_best_only=True)]


# Compile the model
attr.model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# calculate steps based on number of images and batch size
attr.calculate_steps()
attr.increment_seq()

# Persist execution attributes for session resume
save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

# Train the model
history = attr.model.fit_generator(
      attr.train_generator,
      steps_per_epoch=attr.steps_train,
      epochs=attr.epochs,
      validation_data=attr.validation_generator,
      validation_steps=attr.steps_valid,
      callbacks=callbacks,
      verbose=1)

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

write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback)

copy_to_s3(attr)
# os.system("sudo poweroff")
