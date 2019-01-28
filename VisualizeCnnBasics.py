# PURPOSE:
# DCNN visualization tests

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.optimizers import RMSprop, Adam 
from keras.initializers import he_normal
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

from keras.preprocessing import image
from keras import models
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im

# dimensions of our images.
img_width, img_height = 64, 64

testpath='/home/amenegotto/dataset/2d/sem_pre_proc_mini/test/'

img_path = testpath + 'ok/TCGA-K7-A6G5_ff44459fc2d2490a95536285f6d936e1.png'
img = image.load_img(img_path, target_size=(img_width, img_height))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
plt.imshow(img_tensor[0])
plt.show()
plt.clf()

model_fname="/tmp/results/Unimodal/2D/20190128-094939-ckweights.h5"

model = load_model(model_fname)


# Extracts the outputs of the top 12 layers
layer_outputs = [layer.output for layer in model.layers[:12]]

# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 

# Returns a list of numpy arrays, one array per layer activation
activations = activation_model.predict(img_tensor)

# Names of the layers, so you can have them as part of your plot
layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name) 

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
