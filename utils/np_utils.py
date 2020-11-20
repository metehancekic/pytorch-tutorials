import numpy as np


def l1_normalizer(x):
    return x/np.sum(np.abs(x))
