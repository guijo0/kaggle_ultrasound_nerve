import os
import cv2
import time
import glob
import numpy as np
from settings import TRAIN_DATA_PATH, TEST_DATA_PATH
from keras.utils import np_utils

# Data Loading.

def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path, 0)
    return cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)

def load_train(img_rows, img_cols):
    X_train = []
    X_train_id = []
    mask_train = []
    start_time = time.time()

    files = glob.glob(os.path.join(TRAIN_DATA_PATH, '*[0-9].tif'))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_train.append(img)
        X_train_id.append(flbase[:-4])
        mask_path = os.path.join(TRAIN_DATA_PATH, flbase[:-4] + "_mask.tif")
        mask = get_im_cv2(mask_path, img_rows, img_cols)
        mask_train.append(mask)

    print('Read train data time: {} seconds'.format(
        round(time.time() - start_time, 2)))
    data, labels = normalise_data(X_train, mask_train, img_rows, img_cols, 'Train')
    return data, labels, X_train_id

def load_test(img_rows, img_cols):

    files = glob.glob(os.path.join(TEST_DATA_PATH, '*[0-9].tif'))
    X_test = []
    X_test_id = []
    total = 0
    start_time = time.time()
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(flbase[:-4])
        total += 1

    print('Read test data time: {} seconds'.format(
        round(time.time() - start_time, 2)))
    data, _ = normalise_data(X_test, None, img_rows, img_cols)
    return data, _, X_test_id


# Data Preprocessing.

def get_empty_mask_state(mask):
    out = []
    for i in range(len(mask)):
        if mask[i].sum() == 0:
            out.append(0)
        else:
            out.append(1)
    return np.array(out)


def normalise_data(data, labels, img_rows, img_cols, dtype='Test'):

    data = np.array(data, dtype=np.uint8)
    data = data.reshape(data.shape[0], 1, img_rows, img_cols)

    data = data.astype('float32')
    data /= 255

    if labels is not None:
        dtype = 'Training'
        labels = np.array(labels, dtype=np.uint8)
        labels = get_empty_mask_state(labels)
        labels = np_utils.to_categorical(labels, 2)

    print dtype, 'shape:', data.shape
    print dtype, 'samples:', data.shape[0]

    return data, labels
