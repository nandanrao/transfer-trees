import numpy as np
from numba import njit
from collections import namedtuple

# TODO:
#
# 1. Add support for weighting
# 2. Create Criterion for treatment effects

@njit
def mse(X, y):
    # add weights as idx in X and use in Criterion???
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
def np_unique(a):
    prev = a[0]
    unique = [a[0]]
    idxs = [0]
    for i,x in enumerate(a):
        if x != prev:
            unique.append(x)
            idxs.append(i)
        prev = x
    return np.array(unique), np.array(idxs)

@njit
def get_indices(X, p, mn, mx):
    _, idxs = np_unique(X[:, p])
    return idxs[(idxs >= mn) & (idxs <= mx)]


@njit()
def find_threshold(crit, X, y, p, mn, mx):
    N = X.shape[0]

    # bucket for large x? No need to enumerate everything!
    _, base = crit(X,y)
    idxs = get_indices(X, p, mn, mx)
    loss, idx, thresh = np.inf, idxs[0], X[idxs[0], p]

    for i in idxs:
        _, left = crit(X[:i,:], y[:i])
        _, right = crit(X[i:,:], y[i:])

        # should this always be the mean?
        left_share = X[:i].shape[0] / N
        right_share = 1 - left_share
        curr_loss = left*left_share + right*right_share

        if curr_loss < loss:
            loss, idx = curr_loss, i
            thresh = (X[i, p] + X[i-1, p])/2

    return idx, base - loss, thresh

@njit
def sort_for_dim(X, y, p):
    idxs = np.argsort(X[:,p])
    return X[idxs, :], y[idxs]

Split = namedtuple('Split', ['dim', 'idx', 'thresh', 'gain'])
Node = namedtuple('Node', ['dim', 'thresh', 'gain', 'left', 'right'])
Leaf = namedtuple('Leaf', ['prediction', 'score', 'N'])


@njit(parallel=True)
def find_next_split(crit, X, y, dims, min_samples):
    # pick dim with greatest gain (from dims)
    # return dim/threshold/gain

    N,P = X.shape
    indices = np.zeros(P, dtype=np.int64)
    gains, thresholds = np.zeros(P), np.zeros(P)

    # If there are not enough elements to split, stop
    mn, mx = min_samples, N - min_samples
    if mn > mx:
        return None

    for p in dims:
        xi,yi = sort_for_dim(X, y, p)
        i,g,t = find_threshold(crit, xi, yi, p, mn, mx)
        indices[p], gains[p], thresholds[p] = i,g,t

    dim = np.argmax(gains)
    return Split(dim, indices[dim], thresholds[dim], gains[dim])

@njit
def split_data(X, y, split):
    """ returns X_left, y_left, X_right, y_right """
    dim, idx = split.dim, split.idx
    Xo, yo = sort_for_dim(X, y, dim)
    return Xo[:idx, :], yo[:idx], Xo[idx:, :], yo[idx:]


def build_tree(crit, X, y, dims, k, min_gain, min_samples):
    if k == 0:
        return Leaf(*crit(X, y), y.shape[0])

    split = find_next_split(crit, X, y, dims, min_samples)

    if split is None:
        return Leaf(*crit(X, y), y.shape[0])

    Xl, yl, Xr, yr = split_data(X, y, split)
    dim, thresh, gain = split.dim, split.thresh, split.gain

    # stop if one side is empty
    if yl.shape[0] == 0 or yr.shape[0] == 0 or gain < min_gain:
        return Leaf(*crit(X, y), y.shape[0])

    return Node(dim, thresh, gain,
                left = build_tree(crit, Xl, yl, dims, k-1, min_gain, min_samples),
                right = build_tree(crit, Xr, yr, dims, k-1, min_gain, min_samples))


def pick_split(x, node):
    if x[node.dim] >= node.thresh:
        return node.right
    return node.left

def predict(x, node):
    while True:
        try:
            node = pick_split(x, node)
        except AttributeError:
            return node.prediction
