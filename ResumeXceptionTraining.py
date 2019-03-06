# PURPOSE:
# test a structure for resume training after session is lost 

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from TimeCallback import TimeCallback
from ExecutionAttributes import ExecutionAttribute
from Summary import plot_train_stats, create_results_dir, get_base_name, write_summary_txt, save_model
from TrainingResume import save_execution_attributes, read_attributes
import tensorflow as tf
import numpy as np

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)


# Summary Information
SUMMARY_PATH="/mnt/data/fine-tuning/Xception"
#SUMMARY_PATH="c:/temp/results"
#SUMMARY_PATH="/tmp/results"
NETWORK_FORMAT="Unimodal"
IMAGE_FORMAT="2D"
SUMMARY_BASEPATH=create_results_dir(SUMMARY_PATH, NETWORK_FORMAT, IMAGE_FORMAT)

# how many times to execute the training/validation/test
CYCLES = 1

#
# Execution Attributes
INITIAL_EPOCH=1
attr = ExecutionAttribute()
attr.architecture = 'Xception'

results_path = create_results_dir('/mnt/data', 'fine-tuning', attr.architecture)
attr.summ_basename = get_base_name(results_path)
attr.path='/mnt/data/image/2d/com_pre_proc'
attr.set_dir_names()
attr.batch_size = 64  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
attr.epochs = 15

# hyper parameters for model
nb_classes = 2  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
attr.img_width, attr.img_height = 299, 299  # change based on the shape/structure of your images
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation


if K.image_data_format() == 'channels_first':
    input_s = (1, attr.img_width, attr.img_height)
else:
    input_s = (attr.img_width, attr.img_height, 1)

# define model
#attr.model = load_model(attr.summ_basename + '-mid-ckweights.h5')

callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True),
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
class_mode='categorical')

attr.validation_generator = test_datagen.flow_from_directory(
attr.validation_data_dir,
target_size=(attr.img_width, attr.img_height),
batch_size=attr.batch_size,
shuffle=True,
class_mode='categorical')

attr.test_generator = test_datagen.flow_from_directory(
attr.test_data_dir,
target_size=(attr.img_width, attr.img_height),
batch_size=1,
class_mode='categorical',
shuffle=False)

# Persist execution attributes for session resume
save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

# training time
# add the best weights from the train top model
# at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
# we re-load model weights to ensure the best epoch is selected and not the last one.
attr.model = load_model('/mnt/data/fine-tuning/Xception/20190202-122953-mid-ckweights.h5')

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
attr.model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


time_callback = TimeCallback()

# save weights of best training epoch: monitor either val_loss or val_acc
callbacks_list = [time_callback,
    ModelCheckpoint(attr.summ_basename + "-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, verbose=0)
]

# calculate steps based on number of images and batch size
attr.calculate_steps()
attr.increment_seq()


# Persist execution attributes for session resume
save_execution_attributes(attr, attr.summ_basename + '-execution-attributes.properties')

# fine-tune the model
history = attr.model.fit_generator(attr.train_generator,
                    steps_per_epoch=attr.steps_train,
                    epochs=attr.epochs,
                    validation_data=attr.validation_generator,
                    validation_steps=attr.steps_valid,
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
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = attr.model.predict_generator(attr.test_generator, steps=attr.steps_test, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
res="No of errors = {}/{}".format(len(errors), attr.test_generator.samples)
with open(attr.summ_basename + "-predicts.txt", "a") as f:
    f.write(res)
    print(res)
    f.close()

write_summary_txt(attr, "Unimodal", "2D", ['negative', 'positive'])
