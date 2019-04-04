import pandas as pd
import os
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
from ExecutionAttributes import ExecutionAttribute
from keras.preprocessing.image import ImageDataGenerator


def get_patient_info(patient_id, clinical_data):
    patient_row = clinical_data[clinical_data["Patient"] == patient_id]
    return patient_row.drop(['Source', 'Patient', 'Hcc'], axis=1)

missing = []
clinical_data = pd.read_csv('csv/clinical_data.csv')

# images_path='/home/amenegotto/dataset/2d/com_pre_proc/'
images_path='/mnt/data/image/2d/sem_pre_proc/'
print("[INFO] Patients Count: " + str(len(clinical_data)))

print("[INFO] Loading images from " + images_path)

# search recursively for png files
for dirpath, dirs, files in os.walk(images_path):
    images_count = len(files)
    if images_count > 0:
        print("[INFO] Found " + str(len(files)) + " images...")

    for f in files:
        if os.path.splitext(f)[1] == ".png":
            absolute_path = dirpath + '/' + f
            relative_path = absolute_path.replace(images_path, "")
            img_info = relative_path.replace("\\", "/").split("/")

            patient_id = img_info[len(img_info)-1].split('_')[0]
#            print(patient_id)

            patient_data = get_patient_info(patient_id, clinical_data)

            if patient_data.empty:
#                print("Missing patient: " + patient_id)
#                print("Path = " + absolute_path)
                
                if patient_id not in missing:
                    missing.append(patient_id)


for patient in missing:
    print('TCGA-LIHC,' + patient + ',\n')
