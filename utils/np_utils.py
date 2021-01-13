import numpy as np


def l1_normalizer(x):
    return x/np.sum(np.abs(x))


def mean_l1_norm(x):
    norms = np.linalg.norm(x.reshape(x.shape[0], -1), ord=1, axis=-1)
    breakpoint()
    mean_value = np.mean(norms)
    return mean_value


def mean_l2_norm(x):
    norms = np.linalg.norm(x.reshape(x.shape[0], -1), ord=2, axis=-1)
    mean_value = np.mean(norms)
    return mean_value


def mean_linf_norm(x):
    norms = np.max(x, axis=1)
    mean_value = np.mean(norms)
    return mean_value
