# PURPOSE:
# Create a thread safe Custom Batch Generator, initially without image augmentation.

from keras.utils import Sequence
from Datasets import get_image
import numpy as np
from ImageAugmentation import random_shift, random_rotation, random_shear, random_zoom
import random

class MultimodalGenerator(Sequence):
    
    def __init__(self, npy_path, batch_size, height, width, channels, classes, should_shuffle=True, is_debug=False, width_shift=None, height_shift=None, rotation_angle=None, shear_factor=None, min_zoom=None, max_zoom=None):
        self.debug = is_debug

        if self.debug:
            print("On generator init")

        self.dataset = np.load(npy_path)

        if self.debug:
            print("Loaded numpy array " + npy_path)
            print("dataset shape = " + str(self.dataset.shape))

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.n_channels = channels
        self.n_classes = classes
        self.shuffle = should_shuffle
        self.width_shift = width_shift
        self.height_shift = height_shift
        self.rotation_angle = rotation_angle
        self.shear_factor = shear_factor
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.on_epoch_end()

    def __len__(self):
        if self.debug:
           print("_len__")
        return int(np.floor(len(self.dataset) // self.batch_size))

    def on_epoch_end(self):
        if self.debug:
           print("_on_epoch_end__")
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        if self.debug:
            print("_get_item__")
        batch_array = self.dataset[range(index*self.batch_size, (index+1)*self.batch_size)]
        return self.__data_generation(batch_array)

    def __data_generation(self, batch_array):
        if self.debug:
           print("__data_generation__")
           print("len(batch_array) = " + str(len(batch_array)))

        imgs_path = batch_array[:,0]
        batch_img = []
        batch_attributes = batch_array[:,range(1,21)]
        batch_labels = batch_array[:, 21]

        for img_path in imgs_path:
            img = get_image(img_path, self.width, self.height)

            tr = random.randint(0,3)

            if tr = 0 and (self.width_shift is not None or self.height_shift is not None):
                img = random_shift(img, self.width_shift, self.height_shift)
            elif tr = 1 and self.rotation_angle is not None:
                img = random_rotation(img, self.rotation_angle)
            elif tr = 2 and self.shear_factor is not None:
                img = random_shear(img, self.shear_factor)
            elif tr=3 and (self.min_zoom is not None or self.max_zoom is not None):
                img = random_zoom(img, self.min_zoom, self.max_zoom)

            batch_img.append(img)

        X = [(np.array(batch_img, dtype="float") / 255.0), batch_attributes]
        return X, batch_labels