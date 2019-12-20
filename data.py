from collections import namedtuple
from numba import njit

Data = namedtuple('Data', ['W', 'X', 'y'])

def dataset(W, X, y):
    if W is None:
        W = np.ones((X.shape[0], 1))

    return Data(W, X, y)

@njit
def subset_data(dat, start, end):
    return Data(dat.W[start:end, :], dat.X[start:end, :], dat.y[start:end])

@njit
def reindex_data(dat, idx):
    return Data(dat.W[idx, :], dat.X[idx, :], dat.y[idx])
