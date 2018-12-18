import os
import pandas as pd
from shutil import copyfile

# PURPOSE: 
# Reproduce in the filesystem the dataset division (train, validation and test) done in a CSV file
# Don't forget to adjust the CSV header accordingly

#CSV_FILE="/home/amenegotto/Downloads/slices-id.csv"
CSV_FILE="C:/Users/hp/dataset/images/2d/slices-id.csv"
#DST_BASEPATH = "/tmp/"
DST_BASEPATH = "C:/Users/hp/dataset/images/2d"

TRAIN_DIR = "/train"
VALID_DIR = "/valid"
TEST_DIR = "/test"
POSITIVE_DIR = "/ok"
NEGATIVE_DIR = "/nok"


def create_dir(create_dir = False):
    if create_dir: 
        os.makedirs(DST_BASEPATH + POSITIVE_DIR + TRAIN_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + POSITIVE_DIR + VALID_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + POSITIVE_DIR + TEST_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + NEGATIVE_DIR + TRAIN_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + NEGATIVE_DIR + VALID_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + NEGATIVE_DIR + TEST_DIR, exist_ok=True)

create_dir(True)

df = pd.read_csv(CSV_FILE)
for row in df.itertuples():
    print(row)

    src_fname = row.base_path + '/' + row.png_fname
    
    dst_fname = DST_BASEPATH

    if row.hcc_class[:3] == 'POS':
        dst_fname = dst_fname + POSITIVE_DIR
    else:
        dst_fname = dst_fname + NEGATIVE_DIR

    if row.dataset == 'TREIN':
        dst_fname = dst_fname + TRAIN_DIR
    elif row.dataset == 'VALIDA':
        dst_fname = dst_fname + VALID_DIR
    else:
        dst_fname = dst_fname + TEST_DIR     

    dst_fname = dst_fname + '/' + row.png_fname

    copyfile(src_fname, dst_fname)
