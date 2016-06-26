
from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum
import load_data
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss


IMG_ROWS, IMG_COLS = 32, 32
BATCH_SIZE = 32
NB_EPOCH = 100
RANDOM_STATE = 51
NFOLDS = 6


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def getPredScorePercent(train_target, train_id, predictions_valid):
    perc = 0
    for i in range(len(train_target)):
        pred = 1
        if predictions_valid[i][0] > 0.5:
            pred = 0
        real = 1
        if train_target[i][0] > 0.5:
            real = 0
        if real == pred:
            perc += 1
    perc /= len(train_target)
    return perc

train_data, train_target, train_id = load_data.load_train(IMG_ROWS, IMG_COLS)
train_data = train_data.reshape((-1, IMG_ROWS, IMG_COLS, 1))

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

model = conv_2d(model, 256, 2, activation='relu')
model = conv_2d(model, 256, 2, activation='relu')
model = conv_2d(model, 256, 2, activation='relu')
model = max_pool_2d(model, 2, strides=2)

model = fully_connected(model, 4096, activation='relu')
model = dropout(model, 0.5)
model = fully_connected(model, 4096, activation='relu')
model = dropout(model, 0.5)
model = fully_connected(model, 2, activation='softmax')

sgd = Momentum(learning_rate=1e-3, lr_decay=1e-6, momentum=0.9)
model = regression(model, optimizer=sgd,
                   loss='categorical_crossentropy')

model = tflearn.DNN(model, checkpoint_path='convnet_tf_vgg.tfl.ckpt',
                    max_checkpoints=1, tensorboard_verbose=3)

yfull_train = dict()
kf = KFold(len(train_data), n_folds=NFOLDS,
           shuffle=True, random_state=RANDOM_STATE)

num_fold = 0
sum_score = 0
for train_index, test_index in kf:

    X_train, X_valid = train_data[train_index], train_data[test_index]
    Y_train, Y_valid = train_target[train_index], train_target[test_index]

    num_fold += 1

    print('Start KFold number {} from {}'.format(num_fold, NFOLDS))
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))

    model.fit(X_train, Y_train, n_epoch=NB_EPOCH, shuffle=True,
              show_metric=True, batch_size=BATCH_SIZE, snapshot_step=500,
              snapshot_epoch=False, run_id='convnet_tf_vgg', validation_set=(X_valid, Y_valid))

    predictions_valid = model.predict(X_valid)
    score = log_loss(Y_valid, predictions_valid)
    print('Score log_loss: ', score)

    for i in range(len(test_index)):
        yfull_train[test_index[i]] = predictions_valid[i]


predictions_valid = get_validation_predictions(train_data, yfull_train)
score = log_loss(train_target, predictions_valid)
print("Log_loss train independent avg: ", score)

print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(
    score, IMG_ROWS, IMG_COLS, NFOLDS, NB_EPOCH))
perc = getPredScorePercent(train_target, train_id, predictions_valid)
print('Percent success: {}'.format(perc))

model.save("./data/convnet_tf_vgg_256.tfl")
