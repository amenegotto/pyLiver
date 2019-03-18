import pandas as pd
import os
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np


def get_patient_info(patient_id, clinical_data):
    patient_row = clinical_data[clinical_data["Patient"] == patient_id]
    patient_row.drop('Source', axis=1, inplace=True)
    patient_row.drop('Patient', axis=1, inplace=True)
    patient_row.drop('Hcc', axis=1, inplace=True)
    return patient_row


def get_image(image_path, img_width, img_height):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_width, img_height))
    image = img_to_array(image)
    return image


def load_data(images_path, csv_path, img_width, img_height, show_info: False):
    images_train = []
    images_valid = []
    images_test = []
    labels_train = []
    labels_valid = []
    labels_test = []
    attributes_train = []
    attributes_valid = []
    attributes_test = []

    print("[INFO] Loading auxiliary patient data from " + csv_path)

    clinical_data = pd.read_csv(csv_path)

    print("[INFO] Patients Count: " + str(len(clinical_data)))

    print("[INFO] Loading images from " + images_path)

    # search recursively for png files
    for dirpath, dirs, files in os.walk(images_path):
        for f in files:
            if os.path.splitext(f)[1] == ".png":
                absolute_path = dirpath + '/' + f
                relative_path = absolute_path.replace(images_path, "")
                img_info = relative_path.replace("\\", "/").split("/")

                patient_id = img_info[len(img_info)-1].split('_')[0]
                patient_data = get_patient_info(patient_id, clinical_data)

                if img_info[1] == "ok":
                    if img_info[2] == "train":
                        images_train.append(get_image(absolute_path, img_width, img_height))
                        attributes_train.append(patient_data.values)
                        labels_train.append(1)
                    elif img_info[2] == "valid":
                        images_valid.append(get_image(absolute_path, img_width, img_height))
                        attributes_valid.append(patient_data.values)
                        labels_valid.append(1)
                    elif img_info[2] == "test":
                        images_test.append(get_image(absolute_path, img_width, img_height))
                        attributes_test.append(patient_data.values)
                        labels_test.append(1)
                elif img_info[1] == "nok":
                    if img_info[2] == "train":
                        images_train.append(get_image(absolute_path, img_width, img_height))
                        attributes_train.append(patient_data.values)
                        labels_train.append(0)
                    elif img_info[2] == "valid":
                        images_valid.append(get_image(absolute_path, img_width, img_height))
                        attributes_valid.append(patient_data.values)
                        labels_valid.append(0)
                    elif img_info[2] == "test":
                        images_test.append(get_image(absolute_path, img_width, img_height))
                        attributes_test.append(patient_data.values)
                        labels_test.append(0)

    images_train = (np.array(images_train, dtype="float") / 255.0)
    attributes_train = np.array(attributes_train)
    labels_train = np.array(labels_train)
    images_valid = (np.array(images_valid, dtype="float") / 255.0)
    attributes_valid = np.array(attributes_valid)
    labels_valid = np.array(labels_valid)
    images_test = (np.array(images_test, dtype="float") / 255.0)
    attributes_test = np.array(attributes_test)
    labels_test = np.array(labels_test)

    if show_info:
        print("------------------------------------------")
        print("[INFO] Training image count: {:.2f}".format(len(images_train)))
        print("[INFO] Validation image count: {:.2f}".format(len(images_valid)))
        print("[INFO] Testing image count: {:.2f}".format(len(images_test)))
        print("[INFO] Training image size: {:.2f}MB".format(images_train.nbytes / (1024 * 1000.0)))
        print("[INFO] Validation image size: {:.2f}MB".format(images_valid.nbytes / (1024 * 1000.0)))
        print("[INFO] Testing image size: {:.2f}MB".format(images_test.nbytes / (1024 * 1000.0)))
        print("------------------------------------------")

    return images_train, attributes_train, labels_train, images_valid, attributes_valid, labels_valid, images_test, attributes_test, labels_test

