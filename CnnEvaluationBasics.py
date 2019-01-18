import os
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support as score

# Summary Information
SUMMARY_PATH="/tmp/results"
NETWORK_FORMAT="Unimodal"
IMAGE_FORMAT="2D"
SUMMARY_BASEPATH=SUMMARY_PATH + '/' + NETWORK_FORMAT + '/' + IMAGE_FORMAT 



# dimensions of our images.

img_width, img_height = 64, 64

# path = 'C:/Users/hp/Downloads/cars_train'
path='/home/amenegotto/Downloads/cars'
train_data_dir = path + '/trein'
validation_data_dir = path + '/valid'
test_data_dir = path + '/test'
epochs = 2
batch_size = 10 

# Image Generator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=1,
                                                        shuffle=False,
                                                        class_mode='binary')

if K.image_data_format() == 'channels_first':
    input_s = (3, img_width, img_height)
else:
    input_s = (img_width, img_height, 3)

# define model
model = Sequential()
#model.add(Conv2D(64, (3, 3), input_shape=input_s))
#model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3), input_shape=input_s))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(3, 3)))

#model.add(Conv2D(32, (3, 3), input_shape=input_s))
#model.add(Activation('relu'))
#model.add(Conv2D(32, (3, 3), input_shape=input_s))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (4, 4), input_shape=input_s))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), input_shape=input_s))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST=test_generator.samples//test_generator.batch_size

# Train
history = model.fit_generator(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID)

if not os.path.exists(SUMMARY_BASEPATH):
    os.makedirs(SUMMARY_PATH)
    os.makedirs(SUMMARY_PATH + '/' + NETWORK_FORMAT)
    os.makedirs(SUMMARY_PATH + '/' + NETWORK_FORMAT + '/' + IMAGE_FORMAT)


basename=SUMMARY_BASEPATH + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")

# plot history
loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
if len(loss_list) == 0:
    print('Loss is missing in history')
       
## As loss always exists
pepochs = range(1,len(history.history[loss_list[0]]) + 1)
    
## Loss

plt.figure(1)
for l in loss_list:
    plt.plot(pepochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
for l in val_loss_list:
    plt.plot(pepochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
    
## Accuracy
plt.figure(2)
for l in acc_list:
    plt.plot(pepochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
for l in val_acc_list:    
    plt.plot(pepochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(basename + '-training_stats.png')
plt.clf()

with open(basename + ".txt", "a") as f:

    f.write('EXECUTION SUMMARY\n')
    f.write('-----------------\n\n')
    f.write('Network Type: ' + NETWORK_FORMAT + '\n')
    f.write('Image Format: ' + IMAGE_FORMAT + '\n')
    f.write('Image Size: (' + str(img_width) + ',' + str(img_height) + ')\n')
    f.write('Date: ' + datetime.now().strftime("%Y%m%d-%H%M%S")+ '\n')
    f.write('Train Data Path: ' + train_data_dir+ '\n')
    f.write('Train Samples: ' + str(len(train_generator.filenames)) + '\n')
    f.write('Train Steps: ' + str(STEP_SIZE_TRAIN) + '\n')
    f.write('Validation Data Path: ' + validation_data_dir+ '\n')
    f.write('Validation Samples: ' + str(len(validation_generator.filenames)) + '\n')
    f.write('Validation Steps: ' + str(STEP_SIZE_VALID) + '\n')
    f.write('Test Data Path: ' + test_data_dir+ '\n')
    f.write('Test Samples: ' + str(len(test_generator.filenames)) + '\n')
    f.write('Test Steps: ' + str(STEP_SIZE_TEST) + '\n')
    f.write('Epochs: ' + str(epochs) + '\n')
    f.write('Batch Size: ' + str(batch_size) + '\n')

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    print(filenames)
    print(nb_samples)
    f.write("Test Generator Filenames:\n")
    print(filenames, file=f)
    f.write("\nNumber of Test Samples:\n")
    f.write(str(nb_samples) + "\n\n")

    score_gen = model.evaluate_generator(generator=test_generator, steps= nb_samples)

    print(score)
    print('Test Loss:', score_gen[0])
    print('Test accuracy:', score_gen[1])
    f.write('Test Loss:' + str(score_gen[0]) + '\n')
    f.write('Test accuracy:' + str(score_gen[1]) + '\n\n')

    # Confusion Matrix and Classification Report
    test_generator.reset()
    Y_pred = model.predict_generator(test_generator,steps =STEP_SIZE_TEST, verbose=1)
    y_pred = np.rint(Y_pred)

    print(Y_pred)
    print(y_pred)
    print(test_generator.classes)

    f.write('Predicted Values: \n')
    print(Y_pred, file=f)
    f.write('\nRounded Values: \n')
    print(y_pred, file=f)
    f.write('\nClasses: \n')
    print(test_generator.classes, file=f)

    mtx = confusion_matrix(test_generator.classes, y_pred, labels=[1, 0])
    print('Confusion Matrix:')
    print(mtx)
    f.write('\n\nConfusion Matrix:\n')
    f.write('TP   FP\n')
    f.write('TN   FN\n')
    print(mtx, file=f)

    plt.imshow(mtx, cmap='binary', interpolation='None')
    plt.savefig(basename + '-confusion_matrix.png')

    # print('Classification Report')
    target_names = ['barato', 'caro']
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))
    print(classification_report(test_generator.classes, y_pred, target_names=target_names), file=f)

    f.close()


with open(basename + ".csv", "a") as csv:
    csv.write('Test Loss, Test Accuracy, TP, FP, TN, FN, Precision, Recall, F-Score, Support\n')

    precision,recall,fscore,support=score(test_generator.classes, y_pred, average='macro')

    csv.write(str(score_gen[0]) + ',' + str(score_gen[1]) + ',' + str(mtx[0,0]) + ',' + str(mtx[0,1]) + ',' + str(mtx[1,0]) + ',' + str(mtx[1,1]) + ',' + str(precision) + ',' + str(recall) + ',' + str(fscore) + ',' + str(support))
