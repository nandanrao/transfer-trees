from data import Data, reindex_data, stack_data, modify_z, sample_split_data
from numba import njit
import numpy as np
from scipy.stats import gaussian_kde
from collections import namedtuple
from wasserstein import wasserstein_distance
from itertools import combinations

@njit(cache=True)
def _basic_data(X, y, sample_weight):
    N, _ = X.shape

    if sample_weight is None:
        sample_weight = np.ones(N)

    W = sample_weight.reshape(-1, 1)
    W /= W.sum()

    z = np.empty(0, dtype=np.float64)

    return Data(z, W, X, y)

@njit(cache=True)
def _mse(dat):
    y = dat.y
    m = np.mean(y)
    a = (m - y)

    w = dat.W[:, 0]
    score = (a**2).dot(w)
    return m, np.array([score]), np.empty(0, dtype=np.float64)

@njit(cache=True)
def _mae(dat):
    y = dat.y
    med = np.median(y)

    w = dat.W[:, 0]
    score = np.abs(med - y).dot(w)
    return med, np.array([score]), np.empty(0, dtype=np.float64)


def mse(X, y, sample_weight=None):
    return _basic_data(X, y, sample_weight), _basic_data(X, y, sample_weight), _mse

def mae(X, y, sample_weight=None):
    return _basic_data(X, y, sample_weight), _basic_data(X, y, sample_weight), _mae


def kde_score(X):
    scorer = gaussian_kde(X.T).evaluate
    def kde(x):
        return scorer(x.T)
    return kde

@njit(cache=True)
def _normalize(a):
    return a / a.sum()

@njit(cache=True)
def weighted_mean_variance(vals, w):
    nw = _normalize(w)
    mean = vals.dot(nw)
    var = (vals**2).dot(nw) - mean**2
    return mean, var


@njit(cache=True)
def _calc_treatment_stats(w, treatment, y):
    t_vals, c_vals = y[treatment == 1], y[treatment == 0]

    # weighted mean based on weights within leaf
    t_weight, c_weight = w[treatment == 1], w[treatment == 0]

    t_mean, t_var = weighted_mean_variance(t_vals, t_weight)
    c_mean, c_var = weighted_mean_variance(c_vals, c_weight)

    # treatment effect weighted by weight of leaf
    est_treatment_effect = t_mean - c_mean

    t_samples, c_samples = t_vals.shape[0], c_vals.shape[0]

    return est_treatment_effect, t_var, c_var, t_samples, c_samples

@njit(cache=True)
def fill_zip(a, b):
    dif = len(a) - len(b)
    if dif > 0:
        add = np.repeat(b[-1], dif)
        b = np.concatenate((b, add))
    elif dif < 0:
        add = np.repeat(a[-1], abs(dif))
        a = np.concatenate((a, add))

    return list(zip(a, b))

@njit(cache=True)
def pairs_(a):
    b = []
    for i, el in enumerate(a):
        for ell in a[i+1:]:
            b.append((el, ell))
    return b

@njit(cache=True)
def get_ordered_tau(treatment, y):
    t_vals, c_vals = y[treatment == 1], y[treatment == 0]
    t_vals.sort()
    c_vals.sort()
    z = np.array(fill_zip(t_vals, c_vals))
    return z[:, 0] - z[:, 1]

@njit(cache=True)
def _wasserstein_differences(dats):
    taus = [get_ordered_tau(d.W[:, 1], d.y) for d in dats]
    combos = pairs_(np.arange(len(dats)))
    dists = np.array([wasserstein_distance(taus[i], taus[j]) for i, j in combos], dtype=np.float64)

    # TODO: numeric stability? nan out of wassterstein distance?
    dists = np.array([min(1.e5, d) for d in dists], dtype=np.float64)

    # weight based on the minimum of the pair
    weights = np.array([d.W[:, 0].sum() for d in dats])
    weights = np.array([min(weights[i], weights[j]) for i, j in combos])
    weights /= weights.sum()

    # sum based on weights!
    return np.dot(dists, weights)

# @njit(cache=True)
# def _get_vals_weights(dat):
#     y, W = dat.y, dat.W
#     w, treatment, context_idxs = W[:, 0], W[:, 1], W[:, 2]

#     t_vals, c_vals = y[treatment == 1], y[treatment == 0]
#     # weighted mean based on weights within leaf
#     t_weight, c_weight = w[treatment == 1], w[treatment == 0]

#     return (t_vals, t_weight), (c_vals, c_weight)

# @njit(cache=True)
# def _xe(a, b, aw, bw):
#     # TODO: weighted covariance!
#     cov = np.cov(a,b)[1][1]
#     return np.mean(a)*np.mean(b) + cov

# @njit(cache=True)
# def _crosser(da, db):
#     (ta, taw), (ca, caw) = _get_vals_weights(da)
#     (tb, tbw), (cb, cbw) = _get_vals_weights(db)

#     return _xe(ta, tb, taw, tbw) \
#         + _xe(ca, cb, caw, cbw) \
#         - _xe(ta, cb, taw, cbw) \
#         - _xe(tb, ca, tbw, caw)



# build test data and walk through
# this understanding where it goes wrong...
@njit(cache=True)
def _cross_expectations(tau, est_var, dats, ver):
    stats = [_calc_treatment_stats(d.W[:, 0], d.W[:, 1], d.y) for d in dats]
    # threshold???
    stats = [s for s in stats if s[-2] > 1 and s[-1] > 1]

    taus = np.array([t for t, _, _, tn, cn in stats])

    # Sum of variances of individual treatment effects
    # times the 1/K**2 where K is the number of contexts
    var = lambda tv, cv, tn, cn: (tv/tn + cv/cn) - ((tv-cv)**2 )/(tn+cn)

    tau_vars = np.array([var(tv, cv, tn, cn) for _, tv, cv, tn, cn in stats])
    K = len(tau_vars)
    tau_var = tau_vars.sum() / (K**2)

    df = tau_var**2
    df /= np.array([ tv**2 / K**3 for tv in tau_vars]).sum()

    tau_mean = np.mean(taus)

    if ver == 1:
        min_tau = np.min(taus)
        xp = 2 * tau_mean * min_tau - np.mean(tau_vars + taus**2)

    # elif ver == 1:
    #     min_tau = np.min(taus)
    #     xp = 2 * tau * min_tau - (est_var + tau**2)

    # elif ver == 1:
        # min_tau = np.min(taus)
        # xp = 2 * tau * min_tau - np.mean(taus**2)

    elif ver == 2:
        min_tau = np.min(taus)
        xp = 2 * tau_mean * min_tau - np.mean(taus**2)


    elif ver == 3:
        bounds = np.array([t - np.sqrt(tv/tn + cv/cn) for t, tv, cv, tn, cn in stats])
        min_bound = np.min(bounds)
        xp = 2 * (tau_mean - np.sqrt(tau_var)) * min_bound - np.mean(taus**2)

    elif ver == 4:
        xp = _cross_expectations_loo(dats, False)

    elif ver == 5:
        xp = _cross_expectations_loo(dats, True)

    # ver == 5
    else:
        # xx = np.array([(t, tv/tn + cv/cn) for
        #                t, tv, cv, tn, cn in stats])
        inv_vars = 1 / tau_vars
        inv_vars /= inv_vars.sum()
        weighted_tau = np.dot(taus, inv_vars)
        weighted_tau_var = np.dot(tau_vars, inv_vars**2)
        min_tau = np.min(taus)

        xp = 2 * weighted_tau * min_tau - (weighted_tau_var + weighted_tau**2)
        tau_mean = weighted_tau


    return xp, tau_mean, tau_var, df # 8.2


@njit(cache=True)
def _filter(dats, idx):
    return [d for i, d in enumerate(dats) if i != idx]

@njit(cache=True)
def _var(tv, cv, tn, cn):
    return (tv/tn + cv/cn) - ((tv-cv)**2 )/(tn+cn)

@njit(cache=True)
def _pair_loss(dt, dss, inc_var):
    stats_t = _calc_treatment_stats(dt.W[:, 0], dt.W[:, 1], dt.y)
    stats_s = [_calc_treatment_stats(d.W[:, 0], d.W[:, 1], d.y) for d in dss]

    stats_s = [(t, tv, cv, tn, cn)
               for t, tv, cv, tn, cn in stats_s
               if tn > 1 and cn > 1]

    if len(stats_s) == 0:
        return np.nan

    taus_s = np.array([s[0] for s in stats_s])
    vars_s = np.array([_var(tv, cv, tn, cn) for _, tv, cv, tn, cn in stats_s])

    tau_t = stats_t[0]

    score = 2 * (tau_t * np.mean(taus_s)) 
    if inc_var:
        score -= np.mean(vars_s + taus_s**2)
    else:
        score -= np.mean(taus_s**2)
    
    return score


# @njit(cache=True)
# def _pair_loss(dt, ds):
#     stats = [_calc_treatment_stats(d.W[:, 0], d.W[:, 1], d.y)
#              for d in [dt, ds]]

#     means_vars = [(tau, tv/tn + cv/cn)
#                   for tau, tv, cv, tn, cn in stats
#                   if tn > 2 and cn > 2]

#     if len(means_vars) < 2:
#         return np.nan

#     st, ss = means_vars
#     return 2 * (st[0] * ss[0]) - (ss[1] + ss[0]**2)


@njit(cache=True)
def _cross_expectations_loo(dats, inc_var):
    # all dats have the same z
    z = dats[0].z

    if len(dats) == 1:
        # reduces to causal tree loss
        return _pair_loss(dats[0], [dats[0]], inc_var)

    splits = [(dats[i], _filter(dats, i))
              for i in range(len(dats))]

    # for each pair, calc pair loss
    losses = [_pair_loss(t, s, inc_var) for t, s in splits]
    losses = [l for l in losses if not np.isnan(l)]
    
    if len(losses) == 0:
        return -np.inf

    return np.min(np.array(losses))


@njit(cache=True)
def _tau_variances(dats):
    treatments = [_calc_treatment_stats(d.W[:, 0], d.W[:, 1], d.y) for d in dats]
    vals = np.array([t[0] for t in treatments])
    weights = np.array([d.W[:, 0].sum() for d in dats])
    _, var = weighted_mean_variance(vals, weights)
    return var


@njit(cache=True)
def _transfer(dat):
    min_samples, var_weight, importance, target_ctx_idx, xp_version, prediction_mode =\
        dat.z[0], dat.z[1], dat.z[2], dat.z[3], dat.z[4], dat.z[5]
    target_dat = reindex_data(dat, dat.W[:, 2] == target_ctx_idx)
    dat = reindex_data(dat, dat.W[:, 2] != target_ctx_idx)

    W = dat.W
    w, treatment, context_idxs = W[:, 0], W[:, 1], W[:, 2]

    samples_t = treatment.sum()
    samples_c = treatment.shape[0] - samples_t
    # samples_target = target_dat.X.shape[0]
    # or samples_target < min_samples

    if samples_c < min_samples or samples_t < min_samples:
        return np.inf, np.array([np.inf]), np.array([np.inf, np.inf], dtype=np.float64)

    # how can this be moved???
    # hacky to hack into W matrix...
    contexts = np.unique(context_idxs)

    # get treatment effect per context
    # TODO: avoid doing this every time (optimize)
    dats = [reindex_data(dat, context_idxs == i) for i in contexts]

    tau, t_var, c_var, _, _ = _calc_treatment_stats(w, treatment, dat.y)
    est_var = c_var/samples_c + t_var/samples_t

    cross_exp, tau_mean, tau_var, tau_df = _cross_expectations(tau, est_var, dats, xp_version)

    score = -cross_exp

    # cross_exp = _cross_expectations_loo(dats)
    # score = -cross_exp
    # weight...???
    # score = (1 - var_weight) * -cross_exp + (var_weight)*var_pen
    # score *= 2

    if importance:
        target_w = target_dat.W[:, 0]
        score *= target_w.sum() # weight by weights of leaf
    else:
        score *= w.sum()

    if prediction_mode == 0:

        # create df from tau_var...
        # need to figure out how to est degrees
        # of freedom of pooled estimates!!
        pred = tau
        eps = 1e-8
        c_var, t_var = c_var + eps, t_var + eps
        df = est_var**2
        df /= ((c_var**2 / (samples_c**3)) + (t_var**2 / samples_t**3))

    else:
        pred = tau_mean
        est_var = tau_var
        df = tau_df

    # confidence interval
    sd = np.sqrt(est_var)

    return pred, \
        np.array([score, tau**2, 2*est_var, -cross_exp]), \
        np.array([df, sd], dtype=np.float64)


def stack_W(arrays):
    return np.hstack([a.reshape(-1, 1) for a in arrays])

def make_target_data(X, idx):
    N = X.shape[0]
    w = np.ones(N) / N
    W = stack_W([w, np.empty(N), np.repeat(idx, N)])
    return Data(np.array([]), W, X, np.empty(N))



def transfer(X,
             y,
             treatment,
             context_idxs,
             target_X,
             min_samples=1,
             var_weight=0.5,
             target_ctx_idx=-1,
             xp_version=0,
             prediction_mode=0,
             honest=True,
             importance=True):

    sample_weight = np.ones(y.shape[0])
    sample_weight /= sample_weight.sum()

    W = stack_W([sample_weight, treatment, context_idxs])

    z = np.array([min_samples,
                  var_weight,
                  int(importance),
                  target_ctx_idx,
                  xp_version,
                  prediction_mode], dtype=np.float64)

    # make combined data
    source_data = Data(z, W, X, y)
    target_data = make_target_data(target_X, target_ctx_idx)
    data = stack_data([source_data, target_data], source_data.z)

    if honest:
        data, data_est = sample_split_data(data, 0.5, data.W[:, 2], None)
    else:
        data_est = data

    # set min_samples to 1 for estimation est
    data_est = modify_z(data_est, 0, 1)
    return data, data_est, _transfer


@njit(cache=True)
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

    tau, t_var, c_var, _, _ = _calc_treatment_stats(w, treatment, dat.y)

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
    df /= ((c_var**2 / (samples_c**3)) + (t_var**2 / samples_t**3))
    sd = np.sqrt(est_var)

    return tau, np.array([score, tau**2, 2*est_var]), np.array([df, sd], dtype=np.float64)


def causal_tree_criterion(X, y, treatment,
                          sample_weight=None,
                          min_samples=0,
                          honest=True,
                          var_weight=0.5):
    N, _ = X.shape

    if var_weight < 0.0 or var_weight > 1.0:
        raise Exception('var_weight must be between 0.0 and 1.0')

    if sample_weight is None:
        sample_weight = np.ones(N)

    sample_weight /= sample_weight.sum()

    W = np.hstack([sample_weight.reshape(-1, 1),
                   treatment.reshape(-1, 1)])

    z = np.array([min_samples, var_weight], dtype=np.float64)
    data = Data(z, W, X, y)

    if honest:
        data, data_est = sample_split_data(data, 0.5, None, None)
    else:
        data_est = data

    # set min_samples to 1 for estimation est
    data_est = modify_z(data_est, 0, 1)
    return data, data_est, _causal
