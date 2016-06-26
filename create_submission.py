import os
import cv2
import sys
import glob
import datetime
import numpy as np
from settings import TRAIN_DATA_PATH, SUBMISSION_PATH
from utils import img_utils

KOEFF = 0.5

def create_submission(predictions, test_id):
    sub_file = os.path.join(SUBMISSION_PATH,
                            'submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    subm = open(sub_file, "w")
    mask = find_best_mask()
    encode = img_utils.rle_encode(mask)
    subm.write("img,pixels\n")
    for i in range(len(test_id)):
        subm.write(str(test_id[i]) + ',')
        if predictions[i][1] > 0.5:
            subm.write(encode)
        subm.write('\n')
    subm.close()


def find_best_mask():
    files = glob.glob(os.path.join(TRAIN_DATA_PATH, "*_mask.tif"))
    overall_mask = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    overall_mask.fill(0)
    overall_mask = overall_mask.astype(np.float32)

    for fl in files:
        mask = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
        overall_mask += mask
    overall_mask /= 255
    max_value = overall_mask.max()
    overall_mask[overall_mask < KOEFF * max_value] = 0
    overall_mask[overall_mask >= KOEFF * max_value] = 255
    overall_mask = overall_mask.astype(np.uint8)
    return overall_mask

if len(sys.argv) != 2:
    sub_data = 'submission_data.npz'
else:
    sub_data = sys.argv[1]

submission_data = np.load(os.path.join(SUBMISSION_PATH, sub_data))
create_submission(submission_data['test_res'], submission_data['test_id'])
