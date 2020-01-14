from collections import namedtuple
from numba import njit
import numpy as np


Data = namedtuple('Data', ['z', 'W', 'X', 'y'])

def dataset(W, X, y):
    if W is None:
        W = np.ones((X.shape[0], 1))

    z = np.empty(0, dtype=np.float64)

    return Data(z, W, X, y)

@njit
def subset_data(dat, start, end):
    return Data(dat.z, dat.W[start:end, :], dat.X[start:end, :], dat.y[start:end])

@njit
def reindex_data(dat, idx):
    return Data(dat.z, dat.W[idx, :], dat.X[idx, :], dat.y[idx])
