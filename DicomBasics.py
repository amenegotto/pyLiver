import os
import pydicom
from operator import itemgetter

import matplotlib.pyplot as plt

class DcmFile:
    fileName = ""
    order = 0

    def __init__(self, fileName, order):
        self.fileName = fileName
        self.order = order

    def __getitem__(self, item):
        return self.fileName[item]

    def __setitem__(self, key, value):

#ds = pydicom.dcmread("dcm/000001.dcm")
#print(ds.InstanceNumber)
# check for CT
#print(ds.Modality)

# image size
#print(str(ds.Rows) + "x" + str(ds.Columns))

# show image
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
#plt.show(block=True)


images = []
# read to check order
for f in os.listdir("dcm/"):
    ds = pydicom.dcmread("dcm/" + f)
    images.append(DcmFile(f, ds.InstanceNumber))

# order the images
images.sort(key=lambda x:x[1])

for i in images:
    print(i.fileName + " " + str(i.order))



# animate image
#img = None
#for f in os.listdir("dcm/"):
#    ds = pydicom.dcmread("dcm/" + f)
#    if img is None:
#        img = plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
#    else:
#        img.set_data(ds.pixel_array)
#    plt.pause(.1)
#    plt.draw()

#plt.show(block=True)
