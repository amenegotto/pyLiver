import os
import pandas as pd
from shutil import copyfile

# PURPOSE: 
# Reproduce in the filesystem the dataset division (train, validation and test) done in a CSV file
# Don't forget to adjust the CSV header accordingly

CSV_FILE="csv/sem_pre_proc_slices_id.csv"
DST_BASEPATH = "/mnt/data/image/2d/sem_pre_proc"
SRC_DIR = "/mnt/data/image/2d/src"
TRAIN_DIR = "/train"
VALID_DIR = "/valid"
TEST_DIR = "/test"
POSITIVE_DIR = "/ok"
NEGATIVE_DIR = "/nok"


def create_dir(create_dir = False):
    if create_dir: 
        os.makedirs(DST_BASEPATH + TRAIN_DIR + POSITIVE_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + TRAIN_DIR + NEGATIVE_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + VALID_DIR + POSITIVE_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + VALID_DIR + NEGATIVE_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + TEST_DIR + POSITIVE_DIR, exist_ok=True)
        os.makedirs(DST_BASEPATH + TEST_DIR + NEGATIVE_DIR, exist_ok=True)

create_dir(True)

df = pd.read_csv(CSV_FILE)
for row in df.itertuples():
    print(row)

    if SRC_DIR != "":
	src_fname = SRC_DIR + '/' + row.png_fname
    else:
    	src_fname = row.base_path + '/' + row.png_fname

    dst_fname = DST_BASEPATH

    if row.dataset == 'TREIN':
        dst_fname = dst_fname + TRAIN_DIR
    elif row.dataset == 'VALIDA':
        dst_fname = dst_fname + VALID_DIR
    else:
        dst_fname = dst_fname + TEST_DIR     

    if row.hcc_class[:3] == 'POS':
        dst_fname = dst_fname + POSITIVE_DIR
    else:
        dst_fname = dst_fname + NEGATIVE_DIR


    dst_fname = dst_fname + '/' + row.png_fname

    copyfile(src_fname, dst_fname)
