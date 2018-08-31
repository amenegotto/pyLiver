import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

#Numpy basics

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
    
imgs = np.array(image, dtype=np.int16)

# Save Numpy Array
np.save("numpy/000001.npy", imgs)

# Load Numpy Array
imgs_to_process = np.load("numpy/000001.npy").astype(np.float64) 

# Plot Histogram 
plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Plot Numpy Array
img = plt.imshow(imgs_to_process, cmap=plt.cm.bone)
plt.show()

