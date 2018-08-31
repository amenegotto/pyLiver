import os
import pydicom
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

# Convert Dicom images to PNG

MR_SRC='dcm/mr/'
CT_SRC='dcm/ct/'

def Ct2Png(filePath, fileName, outPath):
	print(filePath, ' ', fileName, ' ', outPath)

	ds = pydicom.dcmread(filePath + fileName)
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

	bimg = plt.imshow(imgs, cmap=plt.cm.bone)
	plt.savefig(outPath + os.path.splitext(fileName)[0] + '.png')

def Mri2Png(filePath, fileName, outPath):
	print(filePath, ' ', fileName, ' ', outPath)

	ds = pydicom.dcmread(filePath + fileName)
	image = np.stack(ds.pixel_array)
	imgs = np.array(image, dtype=np.float64)
	bimg = plt.imshow(imgs, cmap=plt.cm.bone)
	plt.savefig(outPath + os.path.splitext(fileName)[0] + '.png')

for f in os.listdir(MR_SRC):
	Mri2Png(MR_SRC, f, 'png/mr/')

for f in os.listdir(CT_SRC):
	Ct2Png(CT_SRC, f, 'png/ct/')

