import os
import pydicom
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

# read image
ds = pydicom.dcmread("dcm/000001.dcm")

# convert to HounsField Scale
image = np.stack(ds.pixel_array)

# Convert to int16 (from sometimes int16), 
# should be possible as values should always be low enough (<32k)
image = image.astype(np.int16)

# Set outside-of-scan pixels to 1
# The intercept is usually -1024, so air is approximately 0
image[image == -2000] = 0

intercept = ds.RescaleIntercept
slope = ds.RescaleSlope
    
if slope != 1:
	image = slope * image.astype(np.float64)
	image = image.astype(np.int16)
        
image += np.int16(intercept)
    
imgs = np.array(image, dtype=np.float64)

# save image before haar transform for comparison
bimg = plt.imshow(imgs, cmap=plt.cm.bone)
plt.savefig('before.png')

# Wavelet 2D Transform
coeffs = pywt.dwt2(imgs, 'haar')

# Wavelet Inverse 2d Transform
filtered = pywt.idwt2(coeffs, 'haar')

# save image after haar transform
aimg = plt.imshow(filtered, cmap=plt.cm.bone)
plt.savefig('after.png')

# compare both in size
print("Before Size =", str(os.stat("before.png").st_size))
print("After  Size = ", str(os.stat("after.png").st_size))

