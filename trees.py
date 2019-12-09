import numpy as np
from numba import njit
from collections import namedtuple

@njit
def mse(X, y):
    m = np.mean(y)
    a = (m - y)
    mse = a.dot(a) / a.shape[0]
    return m, mse

@njit
def mae(X, y):
    med = np.median(y)
    mae = np.mean(np.abs(med - y))
    return med, mae

@njit
def find_threshold(crit, X, y, mn, mx):
    N = X.shape[0]
    a = np.zeros(mx - mn)

    # bucket for large x? No need to enumerate everything,
    _, base = crit(X,y)
    for j,i in enumerate(range(mn, mx)):
        # should this always be the mean?
        _, left = crit(X[:i,:], y[:i])
        _, right = crit(X[i:,:], y[i:])
        a[j] = (left + right) / 2

    gain = base - a
    gidx = np.argmax(gain)
    return gidx + mn, gain[gidx]

@njit
def sort_for_dim(X, y, p):
    idxs = np.argsort(X[:,p])
    return X[idxs, :], y[idxs]

Split = namedtuple('Split', ['dim', 'idx', 'thresh', 'gain'])
Node = namedtuple('Node', ['dim', 'thresh', 'gain', 'left', 'right'])
Leaf = namedtuple('Leaf', ['prediction', 'score'])

@njit
def find_next_split(crit, X, y, dims, min_samples):
    # pick dim with greatest gain (from dims)
    # return dim/threshold/gain

    N,P = X.shape
    indices = np.zeros(P, dtype=np.int64)
    gains, thresholds = np.zeros(P), np.zeros(P)

    # If there are not enough elements to split, stop
    mn, mx = min_samples, N - min_samples + 1
    if mn >= mx:
        return None

    for p in dims:
        xi,yi = sort_for_dim(X, y, p)
        indices[p], gains[p] = find_threshold(crit, xi, yi, mn, mx)
        thresholds[p] = xi[indices[p], p]

    dim = np.argmax(gains)
    idx_max = indices[dim]

    return Split(dim, indices[dim], thresholds[dim], gains[dim])

@njit
def split_data(X, y, split):
    """ returns X_left, y_left, X_right, y_right """
    dim, idx = split.dim, split.idx
    Xo, yo = sort_for_dim(X, y, dim)
    return Xo[:idx, :], yo[:idx], Xo[idx:, :], yo[idx:]


def pick_split(x, node):
    if x[node.dim] >= node.thresh:
        return node.right
    return node.left


def build_tree(crit, X, y, dims, k, min_gain, min_samples):
    if k == 0:
        return Leaf(*crit(X, y))

    split = find_next_split(crit, X, y, dims, min_samples)

    if split is None:
        return Leaf(*crit(X, y))

    Xl, yl, Xr, yr = split_data(X, y, split)
    dim, thresh, gain = split.dim, split.thresh, split.gain

    # stop if one side is empty
    if yl.shape[0] == 0 or yr.shape[0] == 0 or gain < min_gain:
        return Leaf(*crit(X, y))

    return Node(dim, thresh, gain,
                left = build_tree(crit, Xl, yl, dims, k-1, min_gain, min_samples),
                right = build_tree(crit, Xr, yr, dims, k-1, min_gain, min_samples))
