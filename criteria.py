from data import Data, reindex_data
from numba import njit
import numpy as np
from scipy.stats import gaussian_kde
from collections import namedtuple
from wasserstein import wasserstein_distance
from itertools import combinations

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
    return m, np.array([mse]), np.empty(0, dtype=np.float64)

@njit
def _mae(dat):
    y = dat.y
    med = np.median(y)

    w = dat.W[:,0]
    mae = np.abs(med - y).dot(w)
    return med, np.array([mae]), np.empty(0, dtype=np.float64)


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
def weighted_mean_variance(vals, w):
    nw = _normalize(w)
    mean = vals.dot(nw)
    var = (vals**2).dot(nw) - mean**2
    return mean, var

@njit
def _calc_treatment_stats(w, treatment, y):
    t_vals, c_vals = y[treatment == 1], y[treatment == 0]

    # weighted mean based on weights within leaf
    t_weight, c_weight = w[treatment == 1], w[treatment == 0]

    t_mean, t_var = weighted_mean_variance(t_vals, t_weight)
    c_mean, c_var = weighted_mean_variance(c_vals, c_weight)

    # treatment effect weighted by weight of leaf
    est_treatment_effect = t_mean - c_mean

    return est_treatment_effect, t_var, c_var

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
    dists = np.array([wasserstein_distance(taus[i], taus[j]) for i,j in combos], dtype=np.float64)

    # TODO: numeric stability? nan out of wassterstein distance?
    dists = np.array([min(1.e5, d) for d in dists], dtype=np.float64)

    # weight based on the minimum of the pair
    weights = np.array([d.W[:, 0].sum() for d in dats])
    weights = np.array([min(weights[i], weights[j]) for i,j in combos])
    weights /= weights.sum()

    # sum based on weights!
    return np.dot(dists, weights)

@njit
def _tau_variances(dats):
    treatments = [_calc_treatment_stats(d.W[:, 0], d.W[:, 1], d.y) for d in dats]
    vals = np.array([tau for tau,_,_ in treatments])
    weights = np.array([d.W[:, 0].sum() for d in dats])
    _, var = weighted_mean_variance(vals, weights)
    return var


@njit
def _transfer(dat):
    W = dat.W
    w, treatment, context_idxs = W[:, 0], W[:, 1], W[:, 2]
    min_samples = dat.z[0]
    mean_weight, var_weight = dat.z[1], dat.z[2]
    tau_var_weight, wasserstein_weight = dat.z[3], dat.z[4]

    samples_t = treatment.sum()
    samples_c = treatment.shape[0] - samples_t

    if samples_c < min_samples or samples_t < min_samples:
        return np.inf, np.array([np.inf]), np.array([np.inf, np.inf], dtype=np.float64)

    # how can this be moved???
    # hacky to hack into W matrix...
    contexts = np.unique(context_idxs)

    # get treatment effect per context
    # TODO: avoid doing this every time (optimize)
    # Just use a 3-d array for your data!?!? (dat handling would need to support that)
    dats = [reindex_data(dat, context_idxs == i) for i in contexts]

    # penalize treatment effect difference (and variance?) between contexts...
    # make this weighted variance! Put less weight on contexts with few observations.
    tau_var = _tau_variances(dats)

    # also penalize wasserstein distances between different contexts...
    dists = _wasserstein_differences(dats)

    # this variance should be compared to the expected variance, if
    # given the number of observations...
    # tau_var_var = np.var(np.array([vt+vc for _,vt,vc in treatments]))
    # then penalize with another weight!

    tau, t_var, c_var  = _calc_treatment_stats(w, treatment, dat.y)

    est_var = c_var/samples_c + t_var/samples_t
    score = var_weight * 2 * est_var  # penalize 2* estimator variance

    score += tau_var_weight*tau_var # penalize tau_var...
    score += wasserstein_weight*dists # penalize wasserstein dists

    score -= mean_weight * tau**2 # reward squared treatment effect
    score *= w.sum() # weight by weights of leaf

    # confidence interval
    eps = 1e-8
    c_var, t_var = c_var + eps, t_var + eps
    df = est_var**2
    df /= ((c_var**2 / (samples_c**3)) + (t_var**2 / samples_t**3 ))
    sd = np.sqrt(est_var)

    return tau, np.array([score, tau**2, 2*est_var, tau_var]), np.array([df, sd], dtype=np.float64)


def transfer(X,
             y,
             treatment,
             context_idxs,
             target_X,
             min_samples = 1,
             var_weight = 0.25,
             tau_var_weight = 0.25,
             wasserstein_weight = 0.25,
             mean_weight = 0.25,
             importance = True):

    if not np.isclose(var_weight + tau_var_weight + wasserstein_weight + mean_weight, 1.0, 1e-5):
        raise Exception('Transfer criteria needs weights to add to 1')

    ps, pt = kde_score(X), kde_score(target_X)

    sample_weight = np.ones(y.shape[0])
    if importance:
        # TODO: This should not take into account all features, only those
        # we want to consider!!!!!!!!!!!
        # create the weights at split time. This will get expensive,
        # but for now it will have to do...
        # You could do for dimension before splits, then use same weights for
        # each split... renormalized, that's the same actually,
        # "local" weighting is nothing more than renormalization

        # In that case, you need a weight per dimension.
        # created at the beginning
        # W then becomes a... 3D array
        # this doesn't work
        # will need a separate Weight matrix
        # and the whole trees framework needs to take that into account...
        sample_weight = pt(X) / ps(X)

    sample_weight /= sample_weight.sum()

    W = np.hstack([a.reshape(-1, 1) for a in
                   [sample_weight, treatment, context_idxs]])

    z = np.array([min_samples, mean_weight, var_weight, tau_var_weight, wasserstein_weight], dtype=np.float64)

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
        return np.inf, np.array([np.inf]), np.array([np.inf, np.inf], dtype=np.float64)

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

    return tau, np.array([score, tau**2, 2*est_var]), np.array([df, sd], dtype=np.float64)


def causal_tree_criterion(X, y, treatment,
                          sample_weight = None,
                          min_samples = 0,
                          var_weight = 0.5):
    N, P = X.shape

    if var_weight < 0.0 or var_weight > 1.0:
        raise Exception('var_weight must be between 0.0 and 1.0')

    if sample_weight is None:
        sample_weight = np.ones(N)

    sample_weight /= sample_weight.sum()

    W = np.hstack([sample_weight.reshape(-1, 1),
                   treatment.reshape(-1, 1)])

    z = np.array([min_samples, var_weight], dtype=np.float64)

    return Data(z, W, X, y), _causal
