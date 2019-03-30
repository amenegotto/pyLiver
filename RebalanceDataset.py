# PURPOSE: 
# Equalize the original imbalanced dataset based on pre-calculated numbers.
# In the end, tcga_lihc = (tcga_kirp + tcga_stad + cptad_pad) on train, validation and test datasets.

import pandas as pd
import os

REBALANCE_INDEX_PATH='csv/rebalance_index.csv'
SLICES_ID_PATH='csv/sem_pre_proc_slices_id.csv'
BASE_IMG_PATH='/mnt/data/image/2d/sem_pre_proc'
BASE_GARBAGE_PATH='/mnt/data/image/2d/garbage/sem_pre_proc'
NOK_PATH='/nok'
OK_PATH='/ok'
DATASETS_NAMES = [ 'TREIN', 'VALIDA', 'TESTE']
IDX_NAMES = [ 'Train', 'Validation', 'Test']
DATASETS_PATHS = [ '/train', '/valid', '/test']


def create_garbage():
    for i in range(3):
        if not os.path.exists(BASE_GARBAGE_PATH + DATASETS_PATHS[i]):
            os.makedirs(BASE_GARBAGE_PATH + DATASETS_PATHS[i])


def move_to_garbage(i, fname):
    fpath_src = BASE_IMG_PATH + DATASETS_PATHS[i] + NOK_PATH + '/' + fname
    fpath_dst = BASE_GARBAGE_PATH + DATASETS_PATHS[i] + '/' + fname
    if str(os.path.isfile(fpath_src)):
        print('move ' + fpath_src + '=>' + fpath_dst)
        #os.rename(fpath_src, fpath_dst)


def rebalance(study_name, dataset_name, limit, i):
    print('Processing ' + dataset_name + '...')
    qt = 0
    qt_del = 0

    slices_id = pd.read_csv(SLICES_ID_PATH)

    for i_slice, r_slice in slices_id.iterrows():
        
        if study_name in r_slice['base_path']:
           
            if r_slice['dataset'] == dataset_name:
                qt = qt + 1

                if qt > limit:
                    move_to_garbage(i, r_slice['png_fname'])
                    qt_del = qt_del + 1

    print('Limit = ' + str(limit))
    print('Total deleted files = ' + str(qt_del))


create_garbage()

indexes = pd.read_csv(REBALANCE_INDEX_PATH)

for i_idx, r_idx in indexes.iterrows():
    
    if r_idx['Study'] == 'tcga-lihc':
        continue
    
    print('Processing ' + r_idx['Study'] + '...')

    for i in range(3):
        rebalance(r_idx['Study'], DATASETS_NAMES[i], r_idx[IDX_NAMES[i]], i)

    print('---------------')
