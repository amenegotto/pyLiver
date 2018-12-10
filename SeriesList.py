import os
import pydicom
from dcmfile import DcmFile
from operator import itemgetter

# PURPOSE: 
# create a CSV with all paths that has DICOM images

#SRC_DIR='/home/amenegotto/Desktop/tcga-lihc'
SRC_DIRS=['C:/Users/hp/Downloads/tcga-lihc','C:/Users/hp/Downloads/tcga-kirp','C:/Users/hp/Downloads/tcga-stad','C:/Users/hp/Downloads/cptac-pda']

pathlist=[]

# search recursively for dcm directories
for src_dir in SRC_DIRS:
    for dirpath, dirs, files in os.walk(src_dir):
            path = dirpath.split('/')

            for f in files:
                    if os.path.splitext(f)[1] == ".dcm":
                        pathlist.append(dirpath)
                        break


with open("slices.csv", "x") as f:
    for d in pathlist:
        print(d)
        f.write(d + ",,\n")


