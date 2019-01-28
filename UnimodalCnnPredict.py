from keras.models import load_model   
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import numpy as np

model_fname="/tmp/results/Unimodal/2D/20190128-091829-ckweights.h5"
testpath='/home/amenegotto/dataset/2d/sem_pre_proc_mini/test'

model = load_model(model_fname)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
        testpath,
        target_size=(64, 64),
        batch_size=1,
        shuffle=False,
        class_mode='binary',
        color_mode='grayscale')

print(model.evaluate_generator(generator=test_generator, steps=200))


test_generator.reset()
Y_pred = model.predict_generator(test_generator, steps=200, verbose=1)
y_pred = np.rint(Y_pred)

mtx = confusion_matrix(test_generator.classes, y_pred, labels=[1, 0])

print(mtx)
