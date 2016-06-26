import os
import json


with open('./SETTINGS.json') as settings:
    SETTINGS = json.load(settings)

TRAIN_DATA_PATH = SETTINGS["TRAIN_DATA_PATH"]
TEST_DATA_PATH = SETTINGS["TEST_DATA_PATH"]
MODEL_PATH = SETTINGS["MODEL_PATH"]
SUBMISSION_PATH = SETTINGS["SUBMISSION_PATH"]
IMG_ROWS = SETTINGS["IMG_ROWS"]
IMG_COLS = SETTINGS["IMG_COLS"]
