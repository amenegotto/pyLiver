# PURPOSE:
# VGG19 fine tuning for hepatocarcinoma diagnosis through CTs images
# with image augmentation and multimodal inputs

import numpy as np
from keras.applications import VGG19
from keras.layers import Input, concatenate
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Summary import create_results_dir, get_base_name, plot_train_stats, write_summary_txt, copy_to_s3
from ExecutionAttributes import ExecutionAttribute
from TimeCallback import TimeCallback
from TrainingResume import save_execution_attributes
from keras.utils import plot_model
from MultimodalGenerator import MultimodalGenerator


# fix seed for reproducible results (only works on CPU, not GPU)
#seed = 9
#np.random.seed(seed=seed)
#tf.set_random_seed(seed=seed)

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
attr.architecture = 'vgg19'
attr.csv_path = 'csv/clinical_data.csv'
# numpy_path = '/mnt/data/image/2d/numpy/' + IMG_TYPE
attr.numpy_path = '/home/amenegotto/dataset/2d/numpy/' + IMG_TYPE
attr.path = '/mnt/data/image/2d/' + IMG_TYPE

results_path = create_results_dir(SUMMARY_BASEPATH, 'fine-tuning', attr.architecture)
attr.summ_basename = get_base_name(results_path)
attr.set_dir_names()
attr.batch_size = 32
attr.epochs = 1 

attr.img_width = 224
attr.img_height = 224

input_attributes_s = (20,)

#Load the VGG model
vgg_conv = VGG19(weights='imagenet', include_top=False, input_shape=(attr.img_width, attr.img_height, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


flat = Flatten()(vgg_conv.output)

if INTERMEDIATE_FUSION:
    attr.fusion = "Intermediate Fusion"

    attributes_input = Input(shape=input_attributes_s)
    concat = concatenate([flat, attributes_input])

    hidden1 = Dense(1024, activation='relu')(concat)
    drop3 = Dropout(0.30)(hidden1)
    output = Dense(2, activation='softmax')(drop3)

if LATE_FUSION:
    attr.fusion = "Late Fusion"
    hidden1 = Dense(1024, activation='relu')(flat)
    drop3 = Dropout(0.30)(hidden1)
    output_img = Dense(2, activation='softmax')(drop3)
    
    attributes_input = Input(shape=input_attributes_s)
    hidden3 = Dense(32, activation='relu')(attributes_input)
    drop6 = Dropout(0.2)(hidden3)
    hidden4 = Dense(16, activation='relu')(drop6)
    drop7 = Dropout(0.2)(hidden4)
    output_attributes = Dense(1, activation='sigmoid')(drop7)

    concat = concatenate([output_img, output_attributes])
    hidden5 = Dense(4, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(hidden5)

attr.model = Model(inputs=[vgg_conv.input, attributes_input], outputs=output)

plot_model(attr.model, to_file=attr.summ_basename + '-architecture.png')

attr.train_generator = MultimodalGenerator(attr.numpy_path + '/train.npy', attr.batch_size, attr.img_height, attr.img_width, 3, 2, True, False, 0.2, 0.2, 15, 10, 0.2)
attr.validation_generator = MultimodalGenerator(attr.numpy_path + '/valid.npy', attr.batch_size, attr.img_height, attr.img_width, 3, 2, True, False, 0.2, 0.2, 15, 10, 0.2)
attr.test_generator = MultimodalGenerator(attr.numpy_path + 'test.npy', 1, attr.img_height, attr.img_width, 3, 2, True, False)

print("[INFO] Calculating samples and steps...")
attr.calculate_samples_len()

attr.calculate_steps()

attr.increment_seq()

time_callback = TimeCallback()

callbacks = [time_callback, EarlyStopping(monitor='val_acc', patience=15, mode='max', restore_best_weights=True),
             ModelCheckpoint(attr.summ_basename + "-ckweights.h5", mode='max', verbose=1, monitor='val_acc', save_best_only=True)]


# Compile the model
attr.model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

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
      use_multiprocessing=True,
      workers=10,
      verbose=1)

# Save the model
attr.model.save(attr.summ_basename + '-weights.h5')

# Plot train stats
plot_train_stats(history, attr.summ_basename + '-training_loss.png', attr.summ_basename + '-training_accuracy.png')

# Get the filenames from the generator
fnames = attr.fnames_test

# Get the ground truth from generator
ground_truth = attr.labels_test

# Get the predictions from the model using the generator
predictions = attr.model.predict_generator(attr.test_generator, steps=attr.steps_test, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
res = "No of errors = {}/{}".format(len(errors), len(attr.fnames_test))
with open(attr.summ_basename + "-predicts.txt", "a") as f:
    f.write(res)
    print(res)
    f.close()

write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback, callbacks[1].stopped_epoch)

copy_to_s3(attr)
# os.system("sudo poweroff")
