import numpy as np
np.random.seed(2016)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

import load_data

IMG_ROWS, IMG_COLS = 32, 32
BATCH_SIZE = 32
NB_EPOCH = 100
RANDOM_STATE = 51
NFOLDS = 6

model = Sequential()
model.add(Convolution2D(4, 4, 4, border_mode='same', init='he_normal',
                        input_shape = (1, IMG_ROWS, IMG_COLS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Convolution2D(8, 4, 4, border_mode='same', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=0),
]

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


yfull_train = dict()
train_data, train_target, train_id = load_data.load_train(IMG_ROWS, IMG_COLS)
kf = KFold(len(train_data), n_folds=NFOLDS, shuffle=True, random_state=RANDOM_STATE)

num_fold = 0
sum_score = 0
for train_index, test_index in kf:

    X_train, X_valid = train_data[train_index], train_data[test_index]
    Y_train, Y_valid = train_target[train_index], train_target[test_index]

    num_fold += 1

    print('Start KFold number {} from {}'.format(num_fold, NFOLDS))
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))


    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
          shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
          callbacks=callbacks)

    predictions_valid = model.predict(X_valid, batch_size=BATCH_SIZE, verbose=1)
    score = log_loss(Y_valid, predictions_valid)
    print('Score log_loss: ', score)

    for i in range(len(test_index)):
        yfull_train[test_index[i]] = predictions_valid[i]


predictions_valid = get_validation_predictions(train_data, yfull_train)
score = log_loss(train_target, predictions_valid)
print("Log_loss train independent avg: ", score)

print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, IMG_ROWS, IMG_COLS, NFOLDS, NB_EPOCH))
perc = getPredScorePercent(train_target, train_id, predictions_valid)
print('Percent success: {}'.format(perc))

json_string = model.to_json()
open('./data/convnet_keras.json', 'w').write(json_string)
model.save_weights('./data/convnet_keras.h5', overwrite=True)
