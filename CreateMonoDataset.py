import os
import pydicom
import pywt
import numpy as np
from scipy.misc import imsave

# PURPOSE: 
# Create image dataset without pre-processing step (filters, denoising, etc.).
# Just convert DICOM to PNG keeping the patient's reference (path) 

#SRC_DIR='C:/Users/hp/Downloads/tcga-lihc'
SRC_DIR='/home/amenegotto/tcga-lihc'
OUT_DIR='/tmp/tcga-lihc-png'

def Ct2Png(file_path, file_name):
    out_path = file_path.replace(SRC_DIR, OUT_DIR)
    os.makedirs(out_path, exist_ok=True)
    dst_name = out_path + os.path.splitext(file_name)[0] + '.png'
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
           Ct2Png(inputdir, f) 

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


