import os
import pydicom
import numpy as np
from skimage import exposure
import uuid
from scipy.misc import imsave
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure

SRC_DIR = 'C:/Users/hp/Downloads/tcga-lihc/TCGA-LIHC_CT_DCM/TCGA-BC-A3KF/02-18-2002-Outside Read or Comparison BODY CT-18240/2-ABDOMEN ST WO-99980'
OUT_DIR = 'C:/temp/foo'
CSV_FILENAME = 'c:/temp/foo/images-id-with-filters.csv'

def create_image_id(file_path, file_name):
    slice_id = uuid.uuid4().hex
    path = file_path.split(os.sep)

    if 'lihc' in path[0].lower():
        hcc_class = 'POS'
    else:
        hcc_class = 'NEG'

    #    print(path, ',',file_name, ',', slice_id, path[1] + '_' + slice_id + '.png')
    fname = path[1] + '_' + slice_id + '.png'
    with open(CSV_FILENAME, "a") as f:
        f.write(OUT_DIR + ',' + path[1] + ',' + path[2] + ',' + path[
            3] + ',' + file_name + ',' + slice_id + ',' + fname + ',' + hcc_class + ', , \n')
    return fname


def save_png(imgs, file_path, file_name, keep_dir=True):
    if (keep_dir):
        out_path = file_path.replace(SRC_DIR, OUT_DIR)
    else:
        out_path = OUT_DIR + '/'
    os.makedirs(out_path, exist_ok=True)
    dst_name = out_path + create_image_id(file_path, file_name)
    print(file_path, file_name, ' => ', dst_name)

    imsave(dst_name, imgs)


def read_dcm(inputdir, file):
    ds = pydicom.dcmread(inputdir + '/' + file)
    image = np.stack(ds.pixel_array)
    image = image.astype(np.int16)
    image[image == -2000] = 0
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    #return np.array(image, dtype=np.float64)
    return image


def do_adaptative_histogram(img):
    return exposure.equalize_adapthist(img, clip_limit=0.15)


def do_tv_denoise(img):
    return denoise_tv_chambolle(img, weight=0.1, multichannel=False)


def dcm_dir_convert(inputdir):
    for f in os.listdir(inputdir):
        if not os.path.isdir(inputdir + f):
           img_original = read_dcm(inputdir, f)
           img_clahe = do_adaptative_histogram(img_original)
           img_denoise = do_tv_denoise(img_clahe)
           save_png(img_denoise, inputdir, f, False)

dcm_dir_convert(SRC_DIR)
