from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
import numpy as np
from Summary import create_results_dir, get_base_name, plot_train_stats, write_summary_txt
from ExecutionAttributes import ExecutionAttribute

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
#tf.set_random_seed(seed=seed)

# Execution Attributes
attr = ExecutionAttribute()
attr.architecture = 'Inception'

results_path = create_results_dir('/tmp', 'fine-tuning', attr.architecture)
attr.summ_basename = get_base_name(results_path)
attr.path='/home/amenegotto/Downloads/cars'
attr.set_dir_names()
attr.batch_size = 4  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
attr.epochs = 1

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# dimensions of our images.
#Inception input size
attr.img_width, attr.img_height = 299, 299

top_layers_checkpoint_path = 'cp.top.best.hdf5'
fine_tuned_checkpoint_path = 'cp.fine_tuned.best.hdf5'
new_extended_inception_weights = 'final_weights.hdf5'

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
attr.model = Model(inputs=base_model.input, outputs=predictions)

#if os.path.exists(top_layers_checkpoint_path):
#	model.load_weights(top_layers_checkpoint_path)
#	print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
attr.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], )

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
#    shear_range=0.1,
#    zoom_range=0.1,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(
        rescale=1. / 255
        )

attr.train_generator = train_datagen.flow_from_directory(
    attr.train_data_dir,
    target_size=(attr.img_height, attr.img_width),
    batch_size=attr.batch_size,
    class_mode='categorical')

attr.validation_generator = test_datagen.flow_from_directory(
    attr.validation_data_dir,
    target_size=(attr.img_height, attr.img_width),
    batch_size=attr.batch_size,
    class_mode='categorical')

attr.test_generator = test_datagen.flow_from_directory(
    attr.test_data_dir,
    target_size=(attr.img_height, attr.img_width),
    batch_size=1,
    class_mode='categorical')

callbacks_top = [
    ModelCheckpoint(attr.summ_basename + "-mid-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_acc', patience=5, verbose=0)
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
    callbacks=callbacks_top)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

attr.model.load_weights(attr.summ_basename + "-mid-ckweights.h5")

#Save the model after every epoch.
callbacks_list = [
    ModelCheckpoint(attr.summ_basename + "-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, verbose=0)
]

#if os.path.exists(fine_tuned_checkpoint_path):
#	model.load_weights(fine_tuned_checkpoint_path)
#	print ("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in attr.model.layers[:172]:
   layer.trainable = False
for layer in attr.model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
attr.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

history = attr.model.fit_generator(
    attr.train_generator,
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
predictions = attr.model.predict_generator(attr.test_generator, steps=attr.steps_test,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
res="No of errors = {}/{}".format(len(errors),attr.test_generator.samples)
with open(attr.summ_basename + "-predicts.txt", "a") as f:
    f.write(res)
    print(res)
    f.close()

write_summary_txt(attr, "Unimodal", "2D", ['negative', 'positive'])    
