from collections import namedtuple
from numba import njit
import numpy as np
import math

Data = namedtuple('Data', ['z', 'W', 'X', 'y'])

def dataset(W, X, y):
    if W is None:
        W = np.ones((X.shape[0], 1))

    z = np.empty(0, dtype=np.float64)

    return Data(z, W, X, y)

@njit(cache=True)
def subset_data(dat, start, end):
    return Data(dat.z, dat.W[start:end, :], dat.X[start:end, :], dat.y[start:end])

@njit(cache=True)
def reindex_data(dat, idx):
    return Data(dat.z, dat.W[idx, :], dat.X[idx, :], dat.y[idx])

@njit(cache=True)
def sort_for_dim(dat, p):
    idx = np.argsort(dat.X[:, p])
    return reindex_data(dat, idx)


@njit(cache=True)
def _stack_data(a, b, z):
    W = np.vstack((a.W, b.W))
    X = np.vstack((a.X, b.X))
    y = np.concatenate((a.y, b.y))
    return Data(z, W, X, y)


@njit(cache=True)
def stack_data(dats, z):
    da = dats[0]
    for dat in dats[1:]:
        da = _stack_data(da, dat, z)
    return da


@njit(cache=True)
def split_data_by_idx(dat, idx):
    left = subset_data(dat, None, idx)
    right = subset_data(dat, idx, None)
    return left, right


@njit(cache=True)
def modify_z(dat, idx, val):
    z = dat.z.copy()
    z[idx] = val
    return Data(z, dat.W, dat.X, dat.y)

@njit(cache=True)
def _sample_split(dat, size, seed):
    if seed is not None:
        np.random.seed(seed)

    N = dat.y.shape[0]
    s = math.ceil(size * N)
    idxs = np.arange(N)

    np.random.shuffle(idxs)    

    ia, ib = idxs[:s], idxs[s:] 

    return reindex_data(dat, ia), reindex_data(dat, ib)


@njit(cache=True)
def sample_split_data(dat, size, context_idxs=None, seed=None):
    if context_idxs is None:
        return _sample_split(dat, size, seed)

    ids = np.unique(context_idxs)
    da, db = None, None
    for i in ids:
        idx = np.argwhere(i == context_idxs).flatten()
        local_dat = reindex_data(dat, idx)
        a, b = _sample_split(local_dat, size, seed=seed)
        if da is None:
            da, db = a, b
        else:
            da, db = _stack_data(da, a, dat.z), _stack_data(db, b, dat.z)

    return da, db



@njit(cache=True)
def _filter(dats, idx):
    return [d for i, d in enumerate(dats) if i != idx]


@njit(cache=True)
def cv_split_data(dats):
    if len(dats) < 2:
        raise Exception('Cannot perform CV splitting with length < 2')
    return [(dats[i], _filter(dats, i))
            for i in range(len(dats))]



def recursive_split_data(dat, depth, context_idx_loc=None, seed=None):
    if depth == 0:
        return [dat]

    if context_idx_loc:
        context_idxs = dat.W[:, context_idx_loc]
    else:
        context_idxs = None

    a, b = sample_split_data(dat, 0.5, context_idxs, seed)
 
    return recursive_split_data(a, depth - 1, context_idx_loc, seed) + \
        recursive_split_data(b, depth - 1, context_idx_loc, seed)
