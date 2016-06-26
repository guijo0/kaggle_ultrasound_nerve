import numpy as np
np.random.seed(2016)

from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
from models import load_keras_model
from keras.preprocessing.image import ImageDataGenerator

import load_data

IMG_ROWS, IMG_COLS = 32, 32
BATCH_SIZE = 32
NB_EPOCH = 100
RANDOM_STATE = 51
NFOLDS = 6

model = load_keras_model()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=0),
]

train_data, train_target = load_data.load_train(IMG_ROWS, IMG_COLS)
train_data = train_data.reshape(train_data.shape[0], 1, IMG_ROWS, IMG_COLS)

datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True)

kf = KFold(len(train_data), n_folds=NFOLDS, shuffle=True, random_state=RANDOM_STATE)

for num_fold, [train_index, test_index] in kf:

    X_train, X_valid = train_data[train_index], train_data[test_index]
    Y_train, Y_valid = train_target[train_index], train_target[test_index]

    print('Start KFold number {} from {}'.format(num_fold, NFOLDS))

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                        samples_per_epoch=len(X_train), nb_epoch=NB_EPOCH
                        ,validation_data=(X_valid, Y_valid),callbacks=callbacks, verbose=2)


json_string = model.to_json()
open('./data/convnet_keras.json', 'w').write(json_string)
model.save_weights('./data/convnet_keras.h5', overwrite=True)
