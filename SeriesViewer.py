import os
import pydicom
import pandas as pd
from dcmfile import DcmFile
from operator import itemgetter
import subprocess

# PURPOSE: 
# read a CSV with all paths created by SeriesList.py and open directory in MicroDicom Viewer

df = pd.read_csv('slices3.csv')
for d in df:
    print(df['directory'])
    subprocess.run(r"C:\Program Files\MicroDicom\mDicom.exe " + df['directory'])

