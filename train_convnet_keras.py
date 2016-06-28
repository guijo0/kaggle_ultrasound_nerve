import load_data
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping
from models import load_keras_model
from settings import IMG_ROWS, IMG_COLS, BATCH_SIZE, NB_EPOCH, SEED, N_FOLDS

model = load_keras_model()

# Stop when 'val_loss' stops improving over PATIENCE # of Epochs.
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5)
]

train_data, train_target = load_data.load_train(IMG_ROWS, IMG_COLS)
train_data = train_data.reshape(train_data.shape[0], 1, IMG_ROWS, IMG_COLS)

kf = KFold(len(train_data), n_folds=N_FOLDS, shuffle=True, random_state=SEED)

for train_index, test_index in kf:

    X_train, X_valid = train_data[train_index], train_data[test_index]
    Y_train, Y_valid = train_target[train_index], train_target[test_index]

    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
              validation_data=(X_valid, Y_valid), callbacks=callbacks, verbose=2)

json_string = model.to_json()
open('./data/convnet_keras.json', 'w').write(json_string)
model.save_weights('./data/convnet_keras.h5', overwrite=True)
