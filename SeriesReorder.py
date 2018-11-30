import os
import pydicom
from dcmfile import DcmFile
from operator import itemgetter

# PURPOSE: 
# rename directories recursively in dicom ordered
# to make it easy to exclude non-liver related slices

SRC_DIR='/home/amenegotto/Desktop/tcga-lihc'

def dicom_reorder(inputdir, outputdir):
    print('Reordering ' + inputdir)
    images = []
    for f in os.listdir(inputdir):
        if not os.path.isdir(inputdir + f):
            ds = pydicom.dcmread(inputdir + f)
            images.append(DcmFile(f, ds.InstanceNumber))
    s_images = sorted(images, key = lambda x: x.order)

    img = None
    i = 0
    for dcmfile in s_images:
        os.rename(inputdir + dcmfile.fileName, outputdir + 'S' + str(i).zfill(5) + '.dcm')
        i = i + 1



# search recursively for dcm directories
pathlist=[]
for dirpath, dirs, files in os.walk(SRC_DIR):
        path = dirpath.split('/')

        for f in files:
                if os.path.splitext(f)[1] == ".dcm":
                    pathlist.append(dirpath)
                    break


for path in pathlist:
    dicom_reorder(path+'/', path+'/')


