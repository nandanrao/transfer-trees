import numpy as np
from numba import njit
from collections import namedtuple
from data import subset_data, reindex_data
from dataclasses import dataclass
from typing import Union

# TODO:
#
# 1. Add support for weighting
# 2. Create Criterion for treatment effects

Split = namedtuple('Split', ['dim', 'idx', 'thresh', 'gain'])

Leaf = namedtuple('Leaf', ['prediction', 'score', 'N'])

class Leaf():
    def __init__(self, prediction, score, N):
        self.prediction = prediction
        self.score = score
        self.N = N

    def __str__(self, level=1):
        return "   " * (level-1) + "|--" + \
            f'pred: {self.prediction:.4f}, score: {self.score:.4f}, N: {self.N} \n'


class Node():
    def __init__(self, dim, thresh, gain, left, right):
        self.dim = dim
        self.thresh = thresh
        self.gain = gain
        self.left = left
        self.right = right

    def __str__(self, level=1):
        d,t,g = self.dim, self.thresh, self.gain,

        ret = "   " * (level-1) + "|--" + \
              'dim: {0}, thresh: {1:.4f}, gain: {2:.4f} \n'.format(d, t, g)
        for child in [self.left, self.right]:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return str(self)


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
def find_threshold(crit, dat, p, mn, mx):
    X = dat.X
    N, _ = X.shape

    # bucket for large x? No need to enumerate everything!
    _, base = crit(dat)
    idxs = get_indices(X, p, mn, mx)
    loss, idx, thresh = np.inf, mn, X[mn, p]

    for i in idxs:
        _, left = crit(subset_data(dat, None, i))
        _, right = crit(subset_data(dat, i, None))

        # should this always be the mean?
        # left_share = X[:i].shape[0] / N
        # right_share = 1 - left_share
        curr_loss = left + right

        if curr_loss < loss:
            loss, idx = curr_loss, i
            thresh = (X[i, p] + X[i-1, p])/2

    return idx, base - loss, thresh

@njit
def sort_for_dim(dat, p):
    idx = np.argsort(dat.X[:,p])
    return reindex_data(dat, idx)

@njit(parallel=True)
def find_next_split(crit, dat, min_samples):
    # pick dim with greatest gain (from dims)
    # return dim/threshold/gain

    N,P = dat.X.shape
    indices = np.zeros(P, dtype=np.int64)
    gains, thresholds = np.zeros(P), np.zeros(P)

    # If there are not enough elements to split, stop
    mn, mx = min_samples, N - min_samples
    if mn > mx:
        return None

    for p in np.arange(P):
        di = sort_for_dim(dat, p)
        indices[p], gains[p], thresholds[p] = \
            find_threshold(crit, di, p, mn, mx)

    dim = np.argmax(gains)
    return Split(dim, indices[dim], thresholds[dim], gains[dim])


@njit
def split_data(dat, split):
    dim, idx = split.dim, split.idx
    do = sort_for_dim(dat, dim)
    left = subset_data(do, None, idx)
    right = subset_data(do, idx, None)
    return left, right


def build_tree(crit, dat, k, min_gain, min_samples):
    if k == 0:
        return Leaf(*crit(dat), N = dat.y.shape[0])

    split = find_next_split(crit, dat, min_samples)

    if split is None:
        return Leaf(*crit(dat), N = dat.y.shape[0])

    dat_l, dat_r = split_data(dat, split)
    dim, thresh, gain = split.dim, split.thresh, split.gain

    # stop if one side is empty
    if dat_l.y.shape[0] == 0 or dat_r.y.shape[0] == 0 or gain < min_gain:
        return Leaf(*crit(dat), N = dat.y.shape[0])

    return Node(dim, thresh, gain,
                left = build_tree(crit, dat_l, k-1, min_gain, min_samples),
                right = build_tree(crit, dat_r, k-1, min_gain, min_samples))


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

from sklearn.base import BaseEstimator, RegressorMixin

class TransferTreeRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, criterion, max_depth, min_gain = 0.0, min_samples_leaf = 10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y, **kwargs):
        data_, crit = self.criterion(X, y, **kwargs)

        self.tree = build_tree(crit,
                               data_,
                               self.max_depth,
                               self.min_gain,
                               self.min_samples_leaf)

    def predict(self, X):
        return np.array([predict(x, self.tree) for x in X])
