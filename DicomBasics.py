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

#inputDir="/tmp/dcm/"
inputDir="/home/amenegotto/Desktop/tcga-lihc/TCGA-G3-A5SL/09-18-2005-CT ABDOMEN NONENH  ENHANCED-BODY-03197/3-Coronal  cor-09281/"
order=False
images = []

# to view already ordered images...
# REMEMBER: do not trust in the order of os.listdir
i = 0
for i in range(284):
    images.append(DcmFile('S' + str(i).zfill(5)+'.dcm', 0))
s_images=images


# to view unordered images
#for f in os.listdir(inputDir):
#    ds = pydicom.dcmread(inputDir + f)
#    images.append(DcmFile(f, ds.InstanceNumber))
#
#if order:
#    s_images = sorted(images, key = lambda x: x.order)
#else:
#    s_images = images

# animate image
img = None
for f in s_images:
    ds = pydicom.dcmread(inputDir + f.fileName)
    if img is None:
        img = plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    else:
        img.set_data(ds.pixel_array)
    plt.pause(.1)
    plt.draw()

plt.show(block=True)
