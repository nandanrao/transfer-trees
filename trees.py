import numpy as np
from numba import njit
from collections import namedtuple
from data import subset_data, reindex_data
from dataclasses import dataclass
from typing import Union
import scipy.special as sc
from sklearn.base import BaseEstimator, RegressorMixin


Split = namedtuple('Split', ['dim', 'idx', 'thresh', 'gain'])

class Leaf():
    def __init__(self, prediction, score, interval, N):
        self.prediction = prediction
        self.interval = interval
        self.score = score
        self.N = N

    def __str__(self, level=1):
        return "   " * (level-1) + "|--" + \
            f'pred: {self.prediction:.4f}, score: {self.score:.4f}, N: {self.N} \n'

    def __len__(self):
        return 1

class Node():
    def __init__(self, leaf, dim, thresh, left, right, gain = None, tot_gain=None):
        self.leaf = leaf
        self.score = leaf.score
        self.dim = dim
        self.thresh = thresh
        self.gain = gain
        self.tot_gain = tot_gain
        self.left = left
        self.right = right

    def __str__(self, level=1):
        d,t,s,g,tg = self.dim, self.thresh, self.leaf.score, self.gain, self.tot_gain

        ret = "   " * (level-1) + "|--" + \
              'dim: {0}, thresh: {1:.4f}, score: {2:.4f}, gain: {3:.4f}, tot_gain: {4:.4f} \n'.format(d, t, s, g, tg)
        for child in [self.left, self.right]:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return str(self)

    def __len__(self):
        if self.left and self.right:
            return 1 + max([len(c) for c in [self.left, self.right]])
        return 1

    def leaves(self):
        leaves = 0
        for child in [self.left, self.right]:
            if isinstance(child, Node):
                leaves += child.leaves()
            else:
                leaves += 1
        return leaves


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
    vals, idxs = np_unique(X[:, p])
    return idxs[(idxs >= mn) & (idxs <= mx)]


@njit()
def find_threshold(crit, dat, p, mn, mx):
    X = dat.X
    N, _ = X.shape

    # bucket for large x? No need to enumerate everything!
    _, base, __ = crit(dat)
    idxs = get_indices(X, p, mn, mx)
    loss, idx, thresh = np.inf, mn, X[mn, p]

    for i in idxs:
        left_dat, right_dat = subset_data(dat, None, i), subset_data(dat, i, None)
        _, left, __ = crit(left_dat)
        _, right, __ = crit(right_dat)

        curr_loss = left + right

        if curr_loss < loss:
            loss, idx = curr_loss, i
            thresh = (X[i, p] + X[i-1, p])/2

    return idx, base - loss, thresh

@njit
def sort_for_dim(dat, p):
    idx = np.argsort(dat.X[:,p])
    return reindex_data(dat, idx)

@njit()
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
    gain = gains[dim]

    return Split(dim, indices[dim], thresholds[dim], gains[dim])


@njit
def split_data_by_idx(dat, idx):
    left = subset_data(dat, None, idx)
    right = subset_data(dat, idx, None)
    return left, right

@njit
def first_greater(a, thresh):
    for i,x in enumerate(a):
        if x > thresh:
            return a[i], i
    return None

@njit
def split_data_by_thresh(dat, dim, thresh):
    do = sort_for_dim(dat, dim)
    _, idx = first_greater(do.X[:, dim], thresh)
    return split_data_by_idx(do, idx)


def build_tree(crit, dat, k, min_samples):
    pred, score, interval = crit(dat)

    leaf = Leaf(pred, score, interval, N = dat.y.shape[0])

    # Stop if max depth is reached
    if k == 0:
        return leaf

    split = find_next_split(crit, dat, min_samples)

    # Stop if find_next_split fails to find a possible split
    if split is None or split.gain <= 0:
        return leaf

    dat_l, dat_r = split_data_by_thresh(dat, split.dim, split.thresh)
    dim, thresh, gain = split.dim, split.thresh, split.gain

    # stop if one side is empty
    if dat_l.y.shape[0] == 0 or dat_r.y.shape[0] == 0:
        return leaf

    # Depth-first tree building
    left = build_tree(crit, dat_l, k-1, min_samples)
    right = build_tree(crit, dat_r, k-1, min_samples)

    node = Node(leaf, dim, thresh, left, right)
    return node


def pick_split(x, node):
    if x[node.dim] >= node.thresh:
        return node.right
    return node.left

def predict(x, node, interval=None):
    while True:
        try:
            node = pick_split(x, node)
        except AttributeError:
            if interval:
                df, sd = node.interval
                z = sc.stdtrit(df, interval)
                return (node.prediction, node.prediction - z*sd, node.prediction + z*sd)
            return node.prediction


def score_tree(node, dat, crit):
    try:
        dat_l, dat_r = split_data_by_thresh(dat, node.dim, node.thresh)
        left = score_tree(node.left, dat_l, crit)
        right = score_tree(node.right, dat_r, crit)
        return left + right
    except AttributeError:
        _, score, _ = crit(dat)
        return score


from copy import deepcopy


def estimate_tree(node, dat, crit):
    pred, score, interval = crit(dat)

    if np.isinf(pred):
        raise Exception('Estimation of tree failed with infinity value!')

    leaf = Leaf(pred, score, interval, dat.y.shape[0])

    try:
        dat_l, dat_r = split_data_by_thresh(dat, node.dim, node.thresh)
        left = estimate_tree(node.left, dat_l, crit)
        right = estimate_tree(node.right, dat_r, crit)

        node = Node(leaf, node.dim, node.thresh, left, right)

        gain = score - (node.left.score + node.right.score)
        tot_gain = gain
        for child in [left, right]:
            if isinstance(child, Node):
                tot_gain += child.tot_gain

        node.gain, node.tot_gain = gain, tot_gain
        return node

    except AttributeError:
        return leaf


def prune_tree(node, alpha):
    node = deepcopy(node)

    try:
        eff_alpha = node.tot_gain / (node.leaves() - 1)

    except AttributeError:
        # It is a leaf, so just return it
        return node

    if eff_alpha <= alpha:
        return node.leaf

    node.left = prune_tree(node.left, alpha)
    node.right = prune_tree(node.right, alpha)

    # tot_gain needs to change!
    tot_gain = node.gain

    for child in [node.left, node.right]:
        if isinstance(child, Node):
            tot_gain += child.tot_gain

    node.tot_gain = tot_gain

    return node

def get_min_trim(node):
    try:
        alpha = node.tot_gain / (node.leaves() - 1)
        children = [get_min_trim(c) for c in
                    [node.left, node.right]]
        children = [c for c in children if c]
        return min([alpha] + children)
    except AttributeError:
        return None

def _trimmed_trees(node):
    alpha = get_min_trim(node)
    if not alpha:
        return []

    new_tree = prune_tree(node, alpha)
    results = [(alpha, new_tree)]
    return results + _trimmed_trees(new_tree)

def get_trimmed_trees(node):
    return [(-np.inf, node)] + _trimmed_trees(node)


class TransferTreeRegressor(RegressorMixin, BaseEstimator):
    def __init__(self,
                 criterion,
                 max_depth = 10,
                 min_samples_leaf = 2,
                 alpha = None,
                 honest = False):

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.honest = honest

    def fit(self, X, y, **fit_params):
        data_, crit = self.criterion(X, y, **fit_params)

        if self.honest:
            N = y.shape[0]
            S = round(N/2)
            data_, data_est = split_data_by_idx(data_, S)

        tree = build_tree(crit,
                          data_,
                          self.max_depth,
                          self.min_samples_leaf)

        # estimation
        data_, crit = self.criterion(X, y, **{**fit_params, 'min_samples': 0 })

        if self.honest:
            _, data_est = split_data_by_idx(data_, S)
            tree = estimate_tree(tree, data_est, crit)
        else:
            tree = estimate_tree(tree, data_, crit)


        alpha = self.alpha if self.alpha is not None else 0.

        self.og_tree = tree
        self.tree_path = get_trimmed_trees(tree)
        try:
            self.tree = [t for a,t in self.tree_path if a <= alpha][-1]
        except IndexError:
            self.tree = self.tree_path[0][1]

        return self


    def predict(self, X, interval=None):
        return np.array([predict(x, self.tree, interval) for x in X])

    def score(self, X, y, **fit_params):
        data_, crit = self.criterion(X, y, **{**fit_params, 'min_samples': 0 })
        return score_tree(self.tree, data_, crit)
