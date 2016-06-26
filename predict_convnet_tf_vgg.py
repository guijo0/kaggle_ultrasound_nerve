from __future__ import division, print_function, absolute_import

import os
import load_data
import tflearn
import numpy as np

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum
from settings import SUBMISSION_PATH

N_FOLDS = 6
BATCH_SIZE = 32
IMG_ROWS = 32
IMG_COLS = 32

model = input_data(shape=[None, IMG_ROWS, IMG_COLS, 1])

model = conv_2d(model, 4, 4, activation='relu')
model = conv_2d(model, 4, 4, activation='relu')
model = max_pool_2d(model, 2, strides=2)

model = conv_2d(model, 16, 4, activation='relu')
model = conv_2d(model, 16, 4, activation='relu')
model = max_pool_2d(model, 2, strides=2)

model = conv_2d(model, 64, 4, activation='relu')
model = conv_2d(model, 64, 4, activation='relu')
model = conv_2d(model, 64, 4, activation='relu')
model = max_pool_2d(model, 2, strides=2)

model = conv_2d(model, 128, 4, activation='relu')
model = conv_2d(model, 128, 4, activation='relu')
model = conv_2d(model, 128, 4, activation='relu')
model = max_pool_2d(model, 2, strides=2)

model = fully_connected(model, 4096, activation='relu')
model = dropout(model, 0.5)
model = fully_connected(model, 4096, activation='relu')
model = dropout(model, 0.5)
model = fully_connected(model, 2, activation='softmax')

sgd = Momentum(learning_rate=1e-3, lr_decay=1e-6,momentum=0.9)
model = regression(model, optimizer=sgd,
                     loss='categorical_crossentropy')

model = tflearn.DNN(model)

model.load("./data/convnet_tf_vgg.tfl")

# Store test predictions
test_data, _, test_id = load_data.load_test(IMG_ROWS, IMG_COLS)
test_data = test_data.reshape((-1,IMG_ROWS,IMG_COLS, 1))


yfull_test = []
for _ in range(N_FOLDS):
    test_prediction = model.predict(test_data)
    yfull_test.append(test_prediction)


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

test_res = merge_several_folds_mean(yfull_test, N_FOLDS)
np.savez(os.path.join(SUBMISSION_PATH, 'tf_vgg_submission'), test_res=test_res, test_id=test_id)
