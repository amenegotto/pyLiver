from keras.models import load_model   
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import numpy as np

model_fname="/tmp/fine-tuning/vgg19/20190131-104818-ckweights.h5"
testpath='/home/amenegotto/Downloads/cars/test'

model = load_model(model_fname)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
        testpath,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode='categorical',
        #color_mode='grayscale'
        )

print(model.evaluate_generator(generator=test_generator, steps=30))


test_generator.reset()
Y_pred = model.predict_generator(test_generator, steps=30, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = classification_report(true_classes, y_pred, target_names=class_labels)
print(report)

mtx = confusion_matrix(true_classes, y_pred)

print(mtx)
