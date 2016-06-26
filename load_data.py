import os
import cv2
import time
import glob
import numpy as np
from settings import TRAIN_DATA_PATH, TEST_DATA_PATH
from keras.utils import np_utils
from keras.preprocessing.image import Iterator

TIF_FILES = '*[0-9].tif'

def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path, 0)
    return cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)


def load_train(img_rows, img_cols):
    data = []
    labels = []

    files = glob.glob(os.path.join(TRAIN_DATA_PATH, TIF_FILES))

    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        data.append(img)

        mask_path = os.path.join(TRAIN_DATA_PATH, flbase[:-4] + "_mask.tif")
        mask = get_im_cv2(mask_path, img_rows, img_cols)
        labels.append(mask)

    data, labels = normalise_data(data, labels)
    return data, labels

def load_test(img_rows, img_cols):

    files = glob.glob(os.path.join(TEST_DATA_PATH, TIF_FILES))
    X_test = []
    X_test_id = []
    total = 0

    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(flbase[:-4])
        total += 1

    data, _ = normalise_data(X_test)
    return data, X_test_id

def normalise_data(data, labels=None):

    data = np.array(data, dtype=np.uint8).astype('float32')
    data /= 255

    if labels is not None:
        labels = np.array(labels, dtype=np.uint8)
        labels = [1 if sum(l.flatten()) > 0 else 0 for l in labels]
        labels = np_utils.to_categorical(labels, 2)

    return data, labels
