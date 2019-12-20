from data import Data, reindex_data
from numba import njit
import numpy as np
from scipy.stats import gaussian_kde

@njit
def _basic_data(X, y, sample_weight):
    N,P = X.shape

    if sample_weight is None:
        sample_weight = np.ones(N)

    W = sample_weight.reshape(-1, 1)
    W /= W.sum()

    return Data(W, X, y)

@njit
def _mse(dat):
    y = dat.y
    m = np.mean(y)
    a = (m - y)

    w = dat.W[:,0]
    mse = (a**2).dot(w)
    return m, mse

@njit
def _mae(dat):
    y = dat.y
    med = np.median(y)

    w = dat.W[:,0]
    mae = np.abs(med - y).dot(w)
    return med, mae


def mse(X, y, sample_weight = None):
    return _basic_data(X, y, sample_weight), _mse

def mae(X, y, sample_weight = None):
    return _basic_data(X, y, sample_weight), _mae


def kde_score(X):
    scorer = gaussian_kde(X.T).evaluate
    def kde(x):
        return scorer(x.T)
    return kde

@njit
def _normalize(a):
    return a / a.sum()

@njit
def _calc_treatment_stats(w, treatment, y):
    t_vals, c_vals = y[treatment == 1], y[treatment == 0]

    # weighted mean based on weights within leaf
    t_weight, c_weight = w[treatment == 1], w[treatment == 0]

    t_mean, c_mean = t_vals.dot(_normalize(t_weight)), c_vals.dot(_normalize(c_weight))

    # treatment effect weighted by weight of leaf
    est_treatment_effect = t_mean - c_mean

    # penalize the variance of the leaf
    t_var = np.dot((t_vals - t_mean)**2, t_weight / w.sum())
    c_var = np.dot((c_vals - c_mean)**2, c_weight / w.sum())
    var = t_var + c_var

    return est_treatment_effect, var

@njit
def _transfer(dat):
    W = dat.W
    weights, treatment, context_idxs = W[:, 0], W[:, 1], W[:, 2]

    # how can this be moved???
    # hacky to hack into W matrix...
    contexts = np.unique(context_idxs)

    # get treatment effect per context
    dats = [reindex_data(dat, context_idxs == i) for i in contexts]

    treatments = [_calc_treatment_stats(d.W[:, 0], d.W[:, 1], d.y) for d in dats]

    # penalize treatment effect difference (and variance?) between contexts...
    tau_var = np.var(np.array([tau for tau,_ in treatments])) + \
        np.var(np.array([var for _,var in treatments]))

    tau, var = _calc_treatment_stats(weights, treatment, dat.y)

    score = (tau_var + 2*var - tau**2) * weights.sum()

    return tau, score


def transfer(X, y, treatment, context_idxs, target_X):
    ps, pt = kde_score(X), kde_score(target_X)
    sample_weight = pt(X) / ps(X)
    sample_weight /= sample_weight.sum()
    W = np.hstack([a.reshape(-1, 1) for a in
                   [sample_weight, treatment, context_idxs]])
    return Data(W, X, y), _transfer


@njit
def _causal(dat):
    w, treatment = dat.W[:, 0].copy(), dat.W[:, 1]

    tau, var = _calc_treatment_stats(w, treatment, dat.y)

    # variance times 2 - once to adjust for mean,
    # another for variance
    # weight by weights of leaf
    score = (2*var - tau**2) * w.sum()

    return tau, score


def causal_tree_criterion(X, y, treatment, sample_weight = None):
    N, P = X.shape

    if sample_weight is None:
        sample_weight = np.ones(N)

    sample_weight /= sample_weight.sum()

    W = np.hstack([sample_weight.reshape(-1, 1),
                   treatment.reshape(-1, 1)])

    return Data(W, X, y), _causal
