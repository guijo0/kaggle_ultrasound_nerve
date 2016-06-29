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


def tmp_create_more_data():
    tifs = glob.glob(os.path.join(TRAIN_DATA_PATH, TIF_FILES))

    np.random.shuffle(glob.glob(os.path.join(TRAIN_DATA_PATH, TIF_FILES)))
    count = 0
    for img_path in tifs:

        img_base = os.path.basename(img_path)
        img = cv2.imread(img_path, 0)

        mask_path = os.path.join(TRAIN_DATA_PATH, img_base[:-4] + "_mask.tif")
        mask = cv2.imread(mask_path, 0)

        if sum(mask.flatten()) > 0:
            new_img, new_mask = image_transform(img, mask)
            cv2.imwrite(os.path.join(TRAIN_DATA_PATH, "transform" + img_base), new_img)
            cv2.imwrite(os.path.join(TRAIN_DATA_PATH, "transform" + img_base[:-4] + "_mask.tif"), new_mask)
            count += 1
            print count

        if count > 1000:
            break


def image_transform(img, mask):

    rowPix = [i for i, v in enumerate(np.amax(mask, axis=1)) if v > 0]
    colPix = [i for i, v in enumerate(np.amax(mask, axis=0)) if v > 0]

    mi_rp = min(rowPix) - np.random.randint(2,10)
    mi_cp = min(colPix) - np.random.randint(2,10)
    ma_rp = max(rowPix) + np.random.randint(2,10)
    ma_cp = max(colPix) + np.random.randint(2,10)

    w = img.shape[0]
    h = img.shape[1]

    new_image = cv2.resize(img[mi_rp:ma_rp, mi_cp:ma_cp],
                           (h, w), interpolation=cv2.INTER_CUBIC)

    new_mask = cv2.resize(mask[mi_rp:ma_rp, mi_cp:ma_cp],
                          (h, w), interpolation=cv2.INTER_CUBIC)

    r = np.random.random()

    if r > 0.6:
        new_image = cv2.flip(new_image, 1)
        new_mask = cv2.flip(new_mask, 1)
    elif r < 0.3:
        new_image = cv2.flip(new_image, 0)
        new_mask = cv2.flip(new_mask, 0)

    return new_image, new_mask


def normalise_data(data, labels=None):

    data = np.array(data, dtype=np.uint8).astype('float32')
    data /= 255

    if labels is not None:
        labels = np.array(labels, dtype=np.uint8)
        labels = [1 if sum(l.flatten()) > 0 else 0 for l in labels]
        labels = np_utils.to_categorical(labels, 2)

    return data, labels

if __name__ == '__main__':
    tmp_create_more_data()
