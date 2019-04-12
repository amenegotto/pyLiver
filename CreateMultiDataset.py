# PURPOSE:
# create numpy arrays to store all data needed on multimodal networks and save them to disk
# for later reuse on training/validation/test cycle for faster initialization

import numpy as np
from Datasets import create_data_as_numpy, create_data_as_list
from ExecutionAttributes import ExecutionAttribute


attr = ExecutionAttribute()

# dimensions of our images.
attr.img_width, attr.img_height = 96, 96

# network parameters
attr.csv_path = 'csv/clinical_data.csv'

image_type = 'sem_pre_proc/'
npy_path = '/mnt/data/image/2d/numpy/' + image_type
# npy_path = '/home/amenegotto/dataset/2d/numpy/' + image_type
attr.path = '/mnt/data/image/2d/' + image_type
# attr.path = '/home/amenegotto/dataset/2d/' + image_type

attr.set_dir_names()

# First multimodal approach using all dataset as numpy array aka Fat Memory Approach...
#create_data_as_numpy(attr.path, attr.csv_path, attr.img_width, attr.img_height, True, npy_path)

# Third multimodal approach for a thread-safe custom generator
create_data_as_list(attr.path, attr.csv_path, True, npy_path)
