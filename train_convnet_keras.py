import load_data
import numpy as np
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping
from models import load_keras_model
from settings import IMG_ROWS, IMG_COLS, BATCH_SIZE, NB_EPOCH, SEED, N_FOLDS


SINGLE_MODEL = True
MODEL_PATH = './data/convnet_keras''

train_files = np.load('train.npz')
train_data = train_files['arr_0']
train_target = train_files['arr_1']
train_data = train_data.reshape(train_data.shape[0], 1, IMG_ROWS, IMG_COLS)


def save_model(identifier, model):

    json_string = model.to_json()
    model_structure = MODEL_PATH + identifier + '.json'
    open(model_structure, 'w').write(json_string)

    model_weight = MODEL_PATH + identifier + '.h5'
    model.save_weights(model_weight, overwrite=True)


if SINGLE_MODEL:

    model = load_keras_model()
    model.fit(train_data, train_target, batch_size=BATCH_SIZE,
              nb_epoch=NB_EPOCH, verbose=2)
    save_model('SINGLE', model)

else:

    kf = KFold(len(train_data), n_folds=N_FOLDS,
               shuffle=True, random_state=SEED)

    for nfold, [train_index, test_index] in enumerate(kf):

        model = load_keras_model()

        # Stop when 'val_loss' stops improving over PATIENCE # of Epochs.
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5)
        ]

        X_train, X_valid = train_data[train_index], train_data[test_index]
        Y_train, Y_valid = train_target[train_index], train_target[test_index]

        model.fit(X_train, Y_train, batch_size=BATCH_SIZE, validation_data=(
            X_valid, Y_valid), callbacks=callbacks, nb_epoch=NB_EPOCH, verbose=2)

        save_model(str(nfold), model)
