import load_data
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping
from models import load_keras_model
from settings import IMG_ROWS, IMG_COLS, BATCH_SIZE, NB_EPOCH, SEED, N_FOLDS

# Stop when 'val_loss' stops improving over PATIENCE # of Epochs.
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=5)
# ]

train_data, train_target = load_data.load_train(IMG_ROWS, IMG_COLS)
train_data = train_data.reshape(train_data.shape[0], 1, IMG_ROWS, IMG_COLS)

kf = KFold(len(train_data), n_folds=N_FOLDS, shuffle=True, random_state=SEED)

model = load_keras_model(None, None)

# for nfold, [train_index, test_index] in enumerate(kf):

model.fit(train_data, train_target, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[], verbose=2)

json_string = model.to_json()
model_structure = './data/convnet_keras' + str(nfold) + '.json'
model_weight = './data/convnet_keras' + str(nfold) + '.h5'

open(model_structure, 'w').write(json_string)
model.save_weights(model_weight, overwrite=True)
