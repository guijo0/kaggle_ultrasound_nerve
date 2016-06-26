import load_data
import numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD

N_FOLDS = 6
BATCH_SIZE = 32
IMG_ROWS = 32
IMG_COLS = 32

model = model_from_json(open('./data/convnet_keras.json').read())
model.load_weights('./data/convnet_keras.h5')
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

test_data, test_id = load_data.load_test(IMG_ROWS, IMG_COLS)
test_data = test_data.reshape(test_data.shape[0], 1, IMG_ROWS, IMG_COLS)

yfull_test = []
for _ in range(N_FOLDS):
    test_prediction = model.predict(test_data, batch_size=BATCH_SIZE)
    yfull_test.append(test_prediction)

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

test_res = merge_several_folds_mean(yfull_test, N_FOLDS)
np.savez('./output/submission_data', test_res=test_res, test_id=test_id)
