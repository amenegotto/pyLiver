# PURPOSE:
# create numpy arrays to store all data needed on multimodal networks and save them to disk
# for later reuse on training/validation/test cycle for faster initialization

import numpy as np
from Datasets import create_data
from ExecutionAttributes import ExecutionAttribute


attr = ExecutionAttribute()

# dimensions of our images.
attr.img_width, attr.img_height = 96, 96

# network parameters
attr.csv_path = 'csv/clinical_data.csv'

image_type = 'com_pre_proc/'
npy_path = '/mnt/data/numpy/' + image_type
attr.path = '/mnt/data/image/2d/' + image_type

attr.set_dir_names()

images_train, fnames_train, attributes_train, labels_train, images_valid, fnames_valid, attributes_valid, labels_valid, images_test, fnames_test, attributes_test, labels_test = create_data(attr.path, attr.csv_path, attr.img_width, attr.img_height, True)

np.save(npy_path + 'images_train.npy', images_train)
np.save(npy_path + 'fnames_train.npy', fnames_train)
np.save(npy_path + 'attributes_train.npy', attributes_train)
np.save(npy_path + 'labels_train.npy', labels_train)
np.save(npy_path + 'images_valid.npy', images_valid)
np.save(npy_path + 'fnames_valid.npy', fnames_valid)
np.save(npy_path + 'attributes_valid.npy', attributes_valid)
np.save(npy_path + 'labels_valid.npy', labels_valid)
np.save(npy_path + 'images_test.npy', images_test)
np.save(npy_path + 'fnames_test.npy', fnames_test)
np.save(npy_path + 'attributes_test.npy', attributes_test)
np.save(npy_path + 'labels_test.npy', labels_test)
