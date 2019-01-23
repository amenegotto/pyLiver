from keras.models import model_from_json   
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

m = open('/mnt/data/results/Unimodal/2D/20190123-221042-model.json', "rb").read()
model = model_from_json(m) 

model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.0001),
                  metrics=['accuracy'])

model.load_weights('/mnt/data/results/Unimodal/2D/20190123-221042-weights.h5')


test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
        '/mnt/data/image/2d/sem_pre_proc/test',
        target_size=(150, 150),
        batch_size=1,
        shuffle=False,
        class_mode='binary')

print(model.evaluate_generator(generator=test_generator, steps=4683))
