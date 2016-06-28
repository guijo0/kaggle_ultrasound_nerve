from __future__ import division, print_function, absolute_import

import load_data
import numpy as np
from models import load_tflearn_model
from settings import IMG_ROWS, IMG_COLS, BATCH_SIZE, NB_EPOCH, SEED, N_FOLDS

model = load_tflearn_model('./data/convnet_tf_vgg.tfl')

test_data, test_id = load_data.load_test(IMG_ROWS, IMG_COLS)
test_data = test_data.reshape((-1, IMG_ROWS, IMG_COLS, 1))

test_pred = model.predict(test_data)
np.savez('./output/submission_data', test_pred, test_id)
