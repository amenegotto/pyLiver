# PURPOSE:
# Xception fine tuning for hepatocarcinoma diagnosis through CTs images
# with image augmentation and multimodal inputs

from keras.layers import *
from keras.applications import *
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers import Dropout, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Summary import create_results_dir, get_base_name, plot_train_stats, write_summary_txt, copy_to_s3
from ExecutionAttributes import ExecutionAttribute
from TimeCallback import TimeCallback
from TrainingResume import save_execution_attributes
from keras.utils import plot_model
from MultimodalGenerator import MultimodalGenerator

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
attr.architecture = 'Xception'
attr.csv_path = 'csv/clinical_data.csv'
attr.numpy_path = '/mnt/data/image/2d/numpy/' + IMG_TYPE
# attr.numpy_path = '/home/amenegotto/dataset/2d/numpy/' + IMG_TYPE
attr.path = '/mnt/data/image/2d/' + IMG_TYPE

results_path = create_results_dir(SUMMARY_BASEPATH, 'fine-tuning', attr.architecture)
attr.summ_basename = get_base_name(results_path)

attr.set_dir_names()
attr.batch_size = 64  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
attr.epochs = 1

# hyper parameters for model
nb_classes = 2  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
attr.img_width, attr.img_height = 299, 299  # change based on the shape/structure of your images
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum

input_attributes_s = (20,)

# Pre-Trained CNN Model using imagenet dataset for pre-trained weights
base_model = Xception(input_shape=(attr.img_width, attr.img_height, 3), weights='imagenet', include_top=False)

# Top Model Block
glob1 = GlobalAveragePooling2D()(base_model.output)

if INTERMEDIATE_FUSION:
    attr.fusion = "Intermediate Fusion"

    attributes_input = Input(shape=input_attributes_s)
    concat = concatenate([glob1, attributes_input])

    hidden1 = Dense(512, activation='relu')(concat)
    output = Dense(nb_classes, activation='softmax')(hidden1)

if LATE_FUSION:
    attr.fusion = "Late Fusion"
    hidden1 = Dense(512, activation='relu')(glob1)
    output_img = Dense(nb_classes, activation='softmax')(hidden1)

    attributes_input = Input(shape=input_attributes_s)
    hidden3 = Dense(32, activation='relu')(attributes_input)
    drop6 = Dropout(0.2)(hidden3)
    hidden4 = Dense(16, activation='relu')(drop6)
    drop7 = Dropout(0.2)(hidden4)
    output_attributes = Dense(1, activation='sigmoid')(drop7)

    concat = concatenate([output_img, output_attributes])
    hidden5 = Dense(4, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(hidden5)


attr.model = Model(inputs=[base_model.input, attributes_input], outputs=output)

plot_model(attr.model, to_file=attr.summ_basename + '-architecture.png')

attr.train_generator = MultimodalGenerator(attr.numpy_path + '/train.npy', attr.batch_size, attr.img_height, attr.img_width, 3, 2, True, False, 0.2, 0.2, 15, 10, 0.2)
attr.validation_generator = MultimodalGenerator(attr.numpy_path + '/valid.npy', attr.batch_size, attr.img_height, attr.img_width, 3, 2, True, False, 0.2, 0.2, 15, 10, 0.2)
attr.test_generator = MultimodalGenerator(attr.numpy_path + 'test.npy', 1, attr.img_height, attr.img_width, 3, 2, True, False)

print("[INFO] Calculating samples and steps...")
attr.calculate_samples_len()

attr.calculate_steps()

attr.increment_seq()


#print(attr.model.summary())

# # let's visualize layer names and layer indices to see how many layers/blocks to re-train
# # uncomment when choosing based_model_last_block_layer
# for i, layer in enumerate(attr.model.layers):
#     print(i, layer.name)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all layers of the based model that is already pre-trained.
for layer in base_model.layers:
    layer.trainable = False

# save and look at how the data augmentations look like
# save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
# save_prefix='aug',
# save_format='jpeg')

callbacks = [
    ModelCheckpoint(attr.summ_basename + "-mid-ckweights.h5", monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, verbose=0)
]

attr.model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Simple CNN
attr.model.fit_generator(attr.train_generator,
                    steps_per_epoch=attr.steps_train,
                    epochs=attr.epochs,
                    validation_data=attr.validation_generator,
                    validation_steps=attr.steps_valid,
                    use_multiprocessing=True,
                    workers=10,
                    callbacks=callbacks)

# verbose
print("\nStarting to Fine Tune Model\n")

# add the best weights from the train top model
# at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
# we re-load model weights to ensure the best epoch is selected and not the last one.
attr.model.load_weights(attr.summ_basename + "-mid-ckweights.h5")

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
    EarlyStopping(monitor='val_loss', patience=10, verbose=0)
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
                    workers=10,
                    callbacks=callbacks_list)


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

write_summary_txt(attr, NETWORK_FORMAT, IMAGE_FORMAT, ['negative', 'positive'], time_callback, callbacks_list[2].stopped_epoch)

copy_to_s3(attr)
# os.system("sudo poweroff")
