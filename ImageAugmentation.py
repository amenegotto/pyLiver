# PURPOSE:
#   Create a wrapper for keras preprocessing functions, mainly to do exactly the same transformations
#   on MultimodalGenerator as it's done when ImageDataGenerator is used (eg. Unimodal stuff)


from keras.preprocessing.image import random_shift
from keras.preprocessing.image import random_rotation
from keras.preprocessing.image import random_shear
from keras.preprocessing.image import random_zoom


def rnd_shift(img, width_shift, horizontal_shift):
    return random_shift(img, wrg=width_shift, hrg=horizontal_shift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')


def rnd_rotation(img, max_angle):
    return random_rotation(img, max_angle, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')


def rnd_shear(img, factor):
    return random_shear(img, intensity=factor, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')


def rnd_zoom(img, factor):
    return random_zoom(img, zoom_range=(1 - factor, 1 + factor), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
