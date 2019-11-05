from keras.models import load_model

model = load_model('c:/users/hp/downloads/rncpu-hcc.h5')

model.summary()

print('# Layers = ', len(model.layers))