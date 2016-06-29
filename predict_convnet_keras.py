import load_data
import glob
import numpy as np
from settings import IMG_ROWS, IMG_COLS, BATCH_SIZE
from models import load_keras_model

test_data, test_id = load_data.load_test(IMG_ROWS, IMG_COLS)
test_data = test_data.reshape(test_data.shape[0], 1, IMG_ROWS, IMG_COLS)

model = load_keras_model('./data/convnet_keras.json','./data/convnet_keras.h5')

test_pred = model.predict(test_data, batch_size=BATCH_SIZE)

np.savez('./output/submission_data', test_pred, test_id)

# TEST SEVERAL MODELS
# structs = glob.glob('./data/*.json')
# weights = glob.glob('./data/*.h5')
# ?
# predictions = []
# for s, w in zip(structs, weights):
# predictions.append(test_pred)
# np.mean(predictions, axis=0)
# model = load_keras_model(s, w)
