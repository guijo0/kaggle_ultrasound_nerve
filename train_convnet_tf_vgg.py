
from __future__ import division, print_function, absolute_import

import tflearn
import load_data
from sklearn.cross_validation import KFold
from models import load_tflearn_model
from settings import IMG_ROWS, IMG_COLS, BATCH_SIZE, NB_EPOCH, SEED, N_FOLDS

model = load_tflearn_model()

train_data, train_target = load_data.load_train(IMG_ROWS, IMG_COLS)
train_data = train_data.reshape((-1, IMG_ROWS, IMG_COLS, 1))

kf = KFold(len(train_data), n_folds=N_FOLDS, shuffle=True, random_state=SEED)

for train_index, test_index in kf:

    X_train, X_valid = train_data[train_index], train_data[test_index]
    Y_train, Y_valid = train_target[train_index], train_target[test_index]

    model.fit(X_train, Y_train, n_epoch=NB_EPOCH, shuffle=True,
              show_metric=True, batch_size=BATCH_SIZE, snapshot_step=500,
              snapshot_epoch=False, validation_set=(X_valid, Y_valid))

model.save("./data/convnet_tf_krishevsky.tfl")
