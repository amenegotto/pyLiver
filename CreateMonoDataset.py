import os
import pydicom
import pywt
import numpy as np
import uuid
from scipy.misc import imsave

# PURPOSE: 
# Create image dataset without pre-processing step (filters, denoising, etc.).
# Just convert DICOM to PNG creating one UUID for each slice and storing this UUID on a CSV to keep track.
# If needed, it can keep the patient's reference (path) 

SRC_DIR='C:/Users/hp/Downloads/tcga-lihc/TCGA-LIHC_CT_DCM'
OUT_DIR='C:/Users/hp/Downloads/tcga-lihc/tcga-lihc-png'
#SRC_DIR='/home/amenegotto/tcga-lihc'
#OUT_DIR='/tmp/tcga-lihc-png'
CSV_FILENAME='images-id-without-filters.csv'

def create_image_id(file_path, file_name):
    slice_id = uuid.uuid4().hex
    path = file_path.split(os.sep)
#    print(path, ',',file_name, ',', slice_id, path[1] + '_' + slice_id + '.png')
    fname = path[1] + '_' + slice_id + '.png' 
    with open(CSV_FILENAME, "a") as f:
        f.write(', '.join(path) + ',' + file_name + ',' + slice_id + ',' + fname + '\n')
    return fname

def ct_to_png(file_path, file_name, keep_dir = True):
    if (keep_dir):
        out_path = file_path.replace(SRC_DIR, OUT_DIR)
    else:
        out_path = OUT_DIR + '/'
    os.makedirs(out_path, exist_ok=True)
    dst_name = out_path + create_image_id(file_path, file_name)
    print(file_path, file_name, ' => ', dst_name)

    ds = pydicom.dcmread(file_path + file_name)
    image = np.stack(ds.pixel_array)
    image = image.astype(np.int16)
    image[image == -2000] = 0
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    imgs = np.array(image, dtype=np.float64)

    imsave(dst_name, imgs)

def dcm_dir_convert(inputdir):
    for f in os.listdir(inputdir):
        if not os.path.isdir(inputdir + f):
           ct_to_png(inputdir, f, False)
           create_image_id(inputdir, f)

# search recursively for dcm directories
pathlist=[]
for dirpath, dirs, files in os.walk(SRC_DIR):
        path = dirpath.split('/')

        for f in files:
                if os.path.splitext(f)[1] == ".dcm":
                    pathlist.append(dirpath)
                    break


for path in pathlist:
    dcm_dir_convert(path+'/')


