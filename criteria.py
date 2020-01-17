from data import Data, reindex_data
from numba import njit
import numpy as np
from scipy.stats import gaussian_kde
from collections import namedtuple
from wasserstein import wasserstein_distance

@njit
def _basic_data(X, y, sample_weight):
    N,P = X.shape

    if sample_weight is None:
        sample_weight = np.ones(N)

    W = sample_weight.reshape(-1, 1)
    W /= W.sum()

    z = np.empty(0, dtype=np.float64)

    return Data(z, W, X, y)

@njit
def _mse(dat):
    y = dat.y
    m = np.mean(y)
    a = (m - y)

    w = dat.W[:,0]
    mse = (a**2).dot(w)
    return m, mse, np.empty(0, dtype=np.float64)

@njit
def _mae(dat):
    y = dat.y
    med = np.median(y)

    w = dat.W[:,0]
    mae = np.abs(med - y).dot(w)
    return med, mae, np.empty(0, dtype=np.float64)


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
    nt_weight, nc_weight = _normalize(t_weight), _normalize(c_weight)

    t_mean, c_mean = t_vals.dot(nt_weight), c_vals.dot(nc_weight)

    # treatment effect weighted by weight of leaf
    est_treatment_effect = t_mean - c_mean

    # penalize the variance of the leaf
    t_var = (t_vals**2).dot(nt_weight)  - (t_mean)**2
    c_var = (c_vals**2).dot(nc_weight)  - (c_mean)**2

    return est_treatment_effect, t_var, c_var

from itertools import combinations


@njit
def fill_zip(a, b):
    dif = len(a) - len(b)
    if dif > 0:
        add = np.repeat(b[-1], dif)
        b = np.concatenate((b, add))
    elif dif < 0:
        add = np.repeat(a[-1], abs(dif))
        a = np.concatenate((a, add))

    return list(zip(a,b))




@njit
def pairs_(a):
    b = []
    for i,el in enumerate(a):
        for ell in a[i+1:]:
            b.append((el,ell))
    return b


@njit
def get_ordered_tau(treatment, y):
    t_vals, c_vals = y[treatment == 1], y[treatment == 0]
    t_vals.sort()
    c_vals.sort()
    z = np.array(fill_zip(t_vals, c_vals))
    return z[:, 0] - z[:, 1]

@njit
def _wasserstein_differences(dats):
    taus = [get_ordered_tau(d.W[:, 1], d.y) for d in dats]
    combos = pairs_(np.arange(len(dats)))
    dists = np.array([wasserstein_distance(taus[i], taus[j]) for i,j in combos])
    return np.sum(dists)

@njit
def _transfer(dat):
    W = dat.W
    w, treatment, context_idxs = W[:, 0], W[:, 1], W[:, 2]
    min_samples = dat.z[0]
    tau_var_weight, tau_var_var_weight = dat.z[1], dat.z[2]

    samples_t = treatment.sum()
    samples_c = treatment.shape[0] - samples_t

    if samples_c < min_samples or samples_t < min_samples:
        return np.inf, np.inf, np.array([np.inf, np.inf], dtype=np.float64)

    # how can this be moved???
    # hacky to hack into W matrix...
    contexts = np.unique(context_idxs)

    # get treatment effect per context
    # TODO: avoid doing this every time (optimize)
    # Just use a 3-d array for your data!?!? (dat handling would need to support that)
    dats = [reindex_data(dat, context_idxs == i) for i in contexts]



    # penalize treatment effect difference (and variance?) between contexts...
    treatments = [_calc_treatment_stats(d.W[:, 0], d.W[:, 1], d.y) for d in dats]
    tau_var = np.var(np.array([tau for tau,_,_ in treatments]))

    # also penalize wasserstein distances between different contexts...
    dists = _wasserstein_differences(dats)


    # this variance should be compared to the expected variance, if
    # given the number of observations...
    # tau_var_var = np.var(np.array([vt+vc for _,vt,vc in treatments]))

    tau, t_var, c_var  = _calc_treatment_stats(w, treatment, dat.y)

    est_var = c_var/samples_c + t_var/samples_t
    score = 2 * est_var  # penalize 2* estimator variance

    score += tau_var_weight*dists # penalize wasserstein dists
    score += tau_var_var_weight*tau_var # penalize tau_var_var...


    # score += tau_var_weight*tau_var # penalize tau_var...??? how should this be weighted??

    score -= tau**2 # reward squared treatment effect
    score *= w.sum() # weight by weights of leaf

    return tau, score, np.empty(0, dtype=np.float64)


def transfer(X,
             y,
             treatment,
             context_idxs,
             target_X,
             min_samples = 1,
             tau_var_weight = 0.5,
             tau_var_var_weight = 0.5,
             importance = True):

    ps, pt = kde_score(X), kde_score(target_X)

    sample_weight = np.ones(y.shape[0])
    if importance:
        # TODO: This should not take into account all features, only those
        # we want to consider!!!!!!!!!!! (penalty for invariance)
        sample_weight = pt(X) / ps(X)

    sample_weight /= sample_weight.sum()

    W = np.hstack([a.reshape(-1, 1) for a in
                   [sample_weight, treatment, context_idxs]])

    z = np.array([min_samples, tau_var_weight, tau_var_var_weight], dtype=np.float64)

    return Data(z, W, X, y), _transfer

import scipy.special as sc

Interval = namedtuple('interval', ['df', 'sd'])

@njit
def _causal(dat):
    w, treatment = dat.W[:, 0].copy(), dat.W[:, 1]
    min_samples = dat.z[0]
    var_weight = dat.z[1]

    # Controls samples, minimum control and treatment
    # by setting score to infinity if does not satisfy requirements
    # Note: this obviously would make gradient optimization a mess
    samples_t = treatment.sum()
    samples_c = treatment.shape[0] - samples_t

    if samples_c < min_samples or samples_t < min_samples:
        return np.inf, np.inf, np.array([np.inf, np.inf], dtype=np.float64)

    tau, t_var, c_var = _calc_treatment_stats(w, treatment, dat.y)

    # score
    est_var = c_var/samples_c + t_var/samples_t
    score = var_weight * 2 * est_var  # penalize 2* estimator variance
    score -= (1-var_weight) * tau**2 # reward squared treatment effect
    score *= 2 # double to make up for var_weight
    score *= w.sum() # weight by weights of leaf

    # confidence interval
    eps = 1e-8
    c_var, t_var = c_var + eps, t_var + eps
    df = est_var**2
    df /= ((c_var**2 / (samples_c**3)) + (t_var**2 / samples_t**3 ))
    sd = np.sqrt(est_var)

    return tau, score, np.array([df, sd], dtype=np.float64)


def causal_tree_criterion(X, y, treatment,
                          sample_weight = None,
                          min_samples = 0,
                          var_weight = 0.5):
    N, P = X.shape

    if sample_weight is None:
        sample_weight = np.ones(N)

    sample_weight /= sample_weight.sum()

    W = np.hstack([sample_weight.reshape(-1, 1),
                   treatment.reshape(-1, 1)])

    z = np.array([min_samples, var_weight], dtype=np.float64)

    return Data(z, W, X, y), _causal
