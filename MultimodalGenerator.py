# PURPOSE:
# Create a thread safe Custom Batch Generator, initially without image augmentation.

from keras.utils import Sequence
from Datasets import get_image
import numpy as np


class MultimodalGenerator(Sequence):
    
    def __init__(self, npy_path, batch_size, height, width, channels, classes, should_shuffle=True, is_debug=False):
        self.debug = is_debug

        if self.debug:
            print("On generator init")

        self.dataset = np.load(npy_path)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.n_channels = channels
        self.n_classes = classes
        self.shuffle = should_shuffle
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

        imgs_path = self.dataset[:,0]
        batch_img = []
        batch_attributes = self.dataset[:,1]
        batch_labels = self.dataset[:, 2]

        for img_path in imgs_path:
            img = get_image(img_path, self.width, self.height)            
            batch_img.append(img)

        X = [(np.array(batch_img, dtype="float") / 255.0), batch_attributes]
        return X, batch_labels