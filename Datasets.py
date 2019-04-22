# PURPOSE: 
# Utility methods for dataset creation and load

import pandas as pd
import os
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def get_patient_info(patient_id, clinical_data):
    patient_row = clinical_data[clinical_data["Patient"] == patient_id]
    return patient_row.drop(['Source', 'Patient', 'Hcc'], axis=1)


def get_image(image_path, img_width, img_height):
    # print('[INFO] Loading ' + image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_width, img_height))
    image = img_to_array(image)
    return image


def load_numpy(numpy_path, array_name):
    print('[INFO] Loading ' + array_name)
    return np.load(numpy_path + '/' + array_name + '.npy')


def load_data(numpy_path):
    return load_numpy(numpy_path, 'images_train'), load_numpy(numpy_path, 'fnames_train'), load_numpy(numpy_path, 'attributes_train'), load_numpy(numpy_path, 'labels_train'), load_numpy(numpy_path, 'images_valid'), load_numpy(numpy_path, 'fnames_valid'), load_numpy(numpy_path, 'attributes_valid'), load_numpy(numpy_path, 'labels_valid'), load_numpy(numpy_path, 'images_test'), load_numpy(numpy_path, 'fnames_test'), load_numpy(numpy_path, 'attributes_test'), load_numpy(numpy_path, 'labels_test')


def create_data_as_numpy(images_path, csv_path, img_width, img_height, show_info: False, npy_path):
    fnames_train = []
    fnames_valid = []
    fnames_test = []
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
        images_count = len(files)
        if images_count > 0:
            print("[INFO] Found " + str(len(files)) + " images...")

        for f in files:
            if os.path.splitext(f)[1] == ".png":
                absolute_path = dirpath + '/' + f
                relative_path = absolute_path.replace(images_path, "")
                img_info = relative_path.replace("\\", "/").split("/")

                patient_id = img_info[len(img_info)-1].split('_')[0]
#                print(patient_id)

                patient_data = get_patient_info(patient_id, clinical_data)

#                print(patient_data)

                if img_info[1] == "ok":
                    if img_info[0] == "train":
                        images_train.append(get_image(absolute_path, img_width, img_height))
                        attributes_train.append(patient_data.values.ravel())
                        labels_train.append(1)
                        fnames_train.append(relative_path)
                    elif img_info[0] == "valid":
#                        print("valid")
                        images_valid.append(get_image(absolute_path, img_width, img_height))
                        attributes_valid.append(patient_data.values.ravel())
                        labels_valid.append(1)
                        fnames_valid.append(relative_path)
                    elif img_info[0] == "test":
#                        print("test")
                        images_test.append(get_image(absolute_path, img_width, img_height))
                        attributes_test.append(patient_data.values.ravel())
                        labels_test.append(1)
                        fnames_test.append(relative_path)
                elif img_info[1] == "nok":
                    if img_info[0] == "train":
                        images_train.append(get_image(absolute_path, img_width, img_height))
                        attributes_train.append(patient_data.values.ravel())
                        labels_train.append(0)
                        fnames_train.append(relative_path)
                    elif img_info[0] == "valid":
#                        print("valid")
                        images_valid.append(get_image(absolute_path, img_width, img_height))
                        attributes_valid.append(patient_data.values.ravel())
                        labels_valid.append(0)
                        fnames_valid.append(relative_path)
                    elif img_info[0] == "test":
#                        print("test")
                        images_test.append(get_image(absolute_path, img_width, img_height))
                        attributes_test.append(patient_data.values.ravel())
                        labels_test.append(0)
                        fnames_test.append(relative_path)

    np_images_train = (np.array(images_train, dtype="float") / 255.0)
    np_images_valid = (np.array(images_valid, dtype="float") / 255.0)
    np_images_test = (np.array(images_test, dtype="float") / 255.0)

    np_attributes_train = np.array(attributes_train)
    np_attributes_valid = np.array(attributes_valid)
    np_attributes_test = np.array(attributes_test)

    np_labels_train = np.array(labels_train)
    np_labels_valid = np.array(labels_valid)
    np_labels_test = np.array(labels_test)

    np_fnames_train = np.array(fnames_train)
    np_fnames_valid = np.array(fnames_valid)
    np_fnames_test = np.array(fnames_test)

    np.save(npy_path + 'images_train.npy', np_images_train)
    np.save(npy_path + 'images_valid.npy', np_images_valid)
    np.save(npy_path + 'images_test.npy', np_images_test)


    np.save(npy_path + 'fnames_train.npy', np_fnames_train)
    np.save(npy_path + 'fnames_valid.npy', np_fnames_valid)
    np.save(npy_path + 'fnames_test.npy', np_fnames_test)
    
    np.save(npy_path + 'attributes_train.npy', np_attributes_train)
    np.save(npy_path + 'attributes_valid.npy', np_attributes_valid)
    np.save(npy_path + 'attributes_test.npy', np_attributes_test)

    np.save(npy_path + 'labels_train.npy', np_labels_train)
    np.save(npy_path + 'labels_valid.npy', np_labels_valid)
    np.save(npy_path + 'labels_test.npy', np_labels_test)

    if show_info:
        print("------------------------------------------")
        print("[INFO] Training image count: {:.2f}".format(len(np_images_train)))
        print("[INFO] Validation image count: {:.2f}".format(len(np_images_valid)))
        print("[INFO] Testing image count: {:.2f}".format(len(np_images_test)))
        print("[INFO] Training image size: {:.2f}MB".format(np_images_train.nbytes / (1024 * 1000.0)))
        print("[INFO] Validation image size: {:.2f}MB".format(np_images_valid.nbytes / (1024 * 1000.0)))
        print("[INFO] Testing image size: {:.2f}MB".format(np_images_test.nbytes / (1024 * 1000.0)))
        print("------------------------------------------")

    print("Done!")


def create_data_as_list(images_path, csv_path, show_info: False, npy_path, is_categorical):
    train = []
    valid = []
    test = []

    print("[INFO] Loading auxiliary patient data from " + csv_path)

    clinical_data = pd.read_csv(csv_path)

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
                print(absolute_path)
                relative_path = absolute_path.replace(images_path, "")
                img_info = relative_path.replace("\\", "/").split("/")

                patient_id = img_info[len(img_info)-1].split('_')[0]
                print(patient_id)

                patient_data = get_patient_info(patient_id, clinical_data)

#                print(patient_data)
                 
                path_array = [ absolute_path ]
                mid_array = np.hstack((np.array(path_array), patient_data.values.ravel()))

                if img_info[1] == "ok":
                    if is_categorical == False:
                        row = np.hstack((mid_array, np.array([1])))
                    else:
                        row = np.hstack((mid_array, np.array([1, 0])))

                    if img_info[0] == "train":
                        train.append(row)
                    elif img_info[0] == "valid":
                        valid.append(row)
                    elif img_info[0] == "test":
                        test.append(row)
                    else:
                        print("Error: " + absolute_path + " - " + patient_id)
                elif img_info[1] == "nok":
                    if is_categorical == False:
                        row = np.hstack((mid_array, np.array([0])))
                    else:
                        row = np.hstack((mid_array, np.array([0, 1])))

                    if img_info[0] == "train":
                        train.append(row)
                    elif img_info[0] == "valid":
                        valid.append(row)
                    elif img_info[0] == "test":
                        test.append(row)
                    else:
                        print("Error: " + absolute_path + " - " + patient_id)

                else:
                    print("Error: " + absolute_path + " - " + patient_id)

    np_train = np.array(train)
    np_valid = np.array(valid)
    np_test = np.array(test)

    if is_categorical:
        np.save(npy_path + 'train-categorical.npy', np_train)
        np.save(npy_path + 'valid-categorical.npy', np_valid)
        np.save(npy_path + 'test-categorical.npy', np_test)
    else:
        np.save(npy_path + 'train.npy', np_train)
        np.save(npy_path + 'valid.npy', np_valid)
        np.save(npy_path + 'test.npy', np_test)

    if show_info:
        print("------------------------------------------")
        print("[INFO] Training count: {:.2f}".format(len(np_train)))
        print("[INFO] Validation count: {:.2f}".format(len(np_valid)))
        print("[INFO] Testing count: {:.2f}".format(len(np_test)))
        print("[INFO] Training size: {:.2f}MB".format(np_train.nbytes / (1024 * 1000.0)))
        print("[INFO] Validation size: {:.2f}MB".format(np_valid.nbytes / (1024 * 1000.0)))
        print("[INFO] Testing size: {:.2f}MB".format(np_test.nbytes / (1024 * 1000.0)))
        print("------------------------------------------")

    print("Done!")


def show_images(images, cols=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.show()


def multimodal_flow_generator(images, attributes, labels, gen : ImageDataGenerator, bsize, debug=False, gen_seed=666):
    genX1 = gen.flow(images, labels, batch_size=bsize, seed=gen_seed)
    genX2 = gen.flow(images, attributes, batch_size=bsize, seed=gen_seed)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()

            if debug:
                np.testing.assert_array_equal(X1i[0],X2i[0])
                print("\n\n\n\n====================Images==========================")
                show_images(X1i[0], 1)
                print("\n\n====================Attributes==========================")
                print(X2i[1])
                print("\n\n====================Labels==========================")
                print(X1i[1])

            yield [X1i[0], X2i[1]], X1i[1]


def multimodal_flow_from_directory_generator(dir_path, csv_path, gen : ImageDataGenerator, bsize, height, width, clazz_mode, should_shuffle, debug=False, gen_seed=666):
    genX1 = gen.flow_from_directory(dir_path,
                                    shuffle=should_shuffle,
                                    batch_size=bsize,
                                    seed=gen_seed,
                                    class_mode=clazz_mode,
                                    target_size=(width, height))

    attributes = populate_clinical_data(genX1.filenames, csv_path)
    current = 0
    while True:
            if (current + bsize) > len(genX1.filenames):
                current = 0

            X1i = genX1.next()
            
            if X1i[0].shape[0] < bsize:
                X2i = attributes[range(current, current + X1i[0].shape[0])]
                current = current + X1i[0].shape[0]
            else:
                X2i = attributes[range(current, current + bsize)]
                current = current + bsize

            if debug:
                print("\n\n========================Current==========================")
                print("Current = " + str(current))
                print("\n\n\n\n====================Images==========================")
                print(X1i[0].shape)
                # show_images(X1i[0], 1)
                print("\n\n====================Attributes==========================")
                print(X2i)
                print(X2i.shape)
                print("\n\n====================Labels==========================")
                print(X1i[1])
                print(X1i[1].shape)

            yield [X1i[0], X2i], X1i[1]


def populate_clinical_data(filenames, csv_path):
    print("[INFO] Populating clinical data...")
    clinical_attributes = []
    clinical_data = pd.read_csv(csv_path)

    for f in filenames:
        img_info = f.replace("\\", "/").split("/")
        patient_id = img_info[len(img_info) - 1].split('_')[0]
        patient_data = get_patient_info(patient_id, clinical_data)
        clinical_attributes.append(patient_data.values.ravel())

    return np.array(clinical_attributes)


def create_image_generator(should_rescale, training):
    if training:
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=(1. / 255) if should_rescale else None,
            shear_range=10,
            zoom_range=0.2,
            fill_mode='nearest')
    else:
        return ImageDataGenerator(
            rescale=(1. / 255) if should_rescale else None
        )
