import os
import json


with open('./SETTINGS.json') as settings:
    SETTINGS = json.load(settings)

TRAIN_DATA_PATH = SETTINGS["TRAIN_DATA_PATH"]
TEST_DATA_PATH = SETTINGS["TEST_DATA_PATH"]
MODEL_PATH = SETTINGS["MODEL_PATH"]
SUBMISSION_PATH = SETTINGS["SUBMISSION_PATH"]

IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 32
NB_EPOCH = 55
SEED = 51
N_FOLDS = 10
