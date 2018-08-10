import os
import pydicom
from dcmfile import DcmFile
from operator import itemgetter

import matplotlib.pyplot as plt

# read basics
#ds = pydicom.dcmread("dcm/000001.dcm")
#print(ds.InstanceNumber)
# check for CT
#print(ds.Modality)

# image size
#print(str(ds.Rows) + "x" + str(ds.Columns))

# show image
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
#plt.show(block=True)


# order the images
images = []
for f in os.listdir("dcm/"):
    ds = pydicom.dcmread("dcm/" + f)
    images.append(DcmFile(f, ds.InstanceNumber))
s_images = sorted(images, key = lambda x: x.order)

# animate image
img = None
for f in s_images:
    ds = pydicom.dcmread("dcm/" + f.fileName)
    if img is None:
        img = plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    else:
        img.set_data(ds.pixel_array)
    plt.pause(.1)
    plt.draw()

plt.show(block=True)
