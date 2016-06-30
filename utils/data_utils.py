import numpy as np


def load_npy_data(*files):
    return [np.load(filename) for filename in files]
