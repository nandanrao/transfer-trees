import numpy as np
import criteria as c
from criteria import *
from data import dataset

X = lambda N: np.random.normal(size = N).reshape(-1, 1)

def test_mse():

    dat, _, crit, _ = mse(X(8), np.array([10.,10.,10.,10.,30.,30.,30.,30.]))
    assert crit(dat)[1] == 100.
    dat, _, crit, _ = mse(X(4), np.array([10.,10.,10.,10.]))
    assert crit(dat)[1] == 0.
    dat, _, crit, _ = mse(X(8), np.array([5.,5.,5.,5.,10.,10.,10.,10.]))
    assert crit(dat)[1] == 6.25


def test_mse_with_weights():
    W = lambda N: np.array([0.5, 0.25] * int((N/2)))

    dat, _, crit, _ = mse(X(8), np.array([10.,20.,10.,20.,30.,20.,30.,20.]), W(8))
    assert np.isclose(crit(dat)[1], 66.6666666666)

def test_causal():
    dat, _, crit, _ = causal_tree_criterion(X(8),
                                         np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
                                         np.array([0,0,0,0,1,1,1,1]),
                                         honest=False)

    assert crit(dat)[1][0] == -400.

# def test_causal_penalizes_double_variance():
#     dat, _, crit, _ = causal_tree_criterion(X(8),
#                                       np.array([15.,20.,0.,5.,20.,25.,35.,40.]) ,
#                                       np.array([0,0,0,0,1,1,1,1]))

#     assert crit(dat)[1] == -400. + 2*np.var([15., 20., 0., 5.])

# def test_calc_treatment_stats_nails_variance():
#     w = np.ones(4) / 4
#     mean, var = c._calc_treatment_stats(w,
#                                         np.array([0,0,1,1]),
#                                         np.array([10.,20.,30.,20.]))

#     print(mean)
#     print(var)

#     assert False

def test_causal_with_weights_trivial():
    W = lambda N: np.array([0.5, 0.25] * int((N/2)))

    dat, _, crit, _ = causal_tree_criterion(X(8),
                                         np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
                                         np.array([0,0,0,0,1,1,1,1]),
                                         W(8),
                                         honest=False)

    assert np.isclose(crit(dat)[1][0], -400.)

def test_causal_with_weights_outlier():
    W = lambda N: np.array([0.5, 0.25] * int((N/2)))

    dat, _, crit, _ = causal_tree_criterion(X(8),
                                         np.array([10.,10.,10.,10.,30.,30.,30.,300.]),
                                         np.array([0,0,0,0,1,1,1,1]),
                                         np.array([1.,1.,1.,1.,1.,1.,1.,0.1]),
                                         honest=False)

    assert np.isclose(crit(dat)[1][0], 1163., 1.0)


    dat, _, crit, _ = causal_tree_criterion(X(8),
                                         np.array([10.,10.,10.,10.,30.,30.,30.,300.]),
                                         np.array([0,0,0,0,1,1,1,1]),
                                         np.array([1.,1.,1.,1.,1.,1.,1.,0.00001]),
                                         honest=False)

    assert np.isclose(crit(dat)[1][0], -400., 1.0)


def test_causal_with_min_samples_inf_score():
    dat, _, crit, _ = causal_tree_criterion(X(8),
                                         np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
                                         np.array([0,0,0,0,1,1,1,1]),
                                         min_samples = 5,
                                         honest=False)

    assert crit(dat)[1][0] == np.inf

def test_get_ordered_tau():
    ordered = get_ordered_tau(np.array([0,0,0,0,1,1,1,1]),
                              np.array([15.,10.,20.,25.,45., 35., 40., 30.]))

    assert np.all(ordered == np.array([20, 20, 20, 20]))

def test_get_ordered_tau_uneven_arrays():
    ordered = get_ordered_tau(np.array([0,0,0,0,1,1,1,1,1]),
                              np.array([15.,10.,20.,25.,45., 35., 40., 30., 45.]))

    assert np.all(ordered == np.array([20, 20, 20, 20, 20]))

def test_wasserstein_differences_same():
    ys = [np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
          np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
          np.array([10.,10.,10.,10.,30.,30.,30.,30.])]

    treatment = np.array([0,0,0,0,1,1,1,1])
    W = np.vstack([np.ones(8), treatment]).T

    dats = [dataset(W, X(8), ys[i]) for i in range(3)]
    dist = c._wasserstein_differences(dats)
    assert np.isclose(dist, 0.0, 1e-6)

def test_wasserstein_differences_different():
    ys = [np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
          np.array([20.,20.,20.,20.,30.,30.,30.,30.]),
          np.array([30.,30.,30.,30.,30.,30.,30.,30.])]

    treatment = np.array([0,0,0,0,1,1,1,1,])
    W = np.vstack([np.ones(8), treatment]).T

    dats = [dataset(W, X(8), ys[i]) for i in range(3)]
    dist = c._wasserstein_differences(dats)
    assert np.isclose(dist, 13.3333, 1e-3)

def test_wasserstein_differences_weighted_small():
    ys = [np.array([20.,20.,20.,20.,20.,20.,30.,30.,30.,30.,30.,30.]),
          np.array([20.,20.,20.,20.,20.,20.,30.,30.,30.,30.,30.,30.]),
          np.array([40., 60.])
    ]

    treatment = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
    W = np.vstack([np.ones(12), treatment]).T
    W2 = np.vstack([np.ones(2), np.array([0,1])]).T

    dats = [dataset(W, X(12), ys[0]),
            dataset(W, X(12), ys[1]),
            dataset(W2, X(2), ys[2])]

    dist = c._wasserstein_differences(dats)
    assert np.isclose(dist, 2.5, 1e-3)

def test_wasserstein_differences_weighted_larger():
    ys = [np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
          np.array([20.,20.,20.,20.,30.,30.,30.,30.]),
          np.array([40., 60.])
    ]
    treatment = np.array([0,0,0,0,1,1,1,1])
    W = np.vstack([np.ones(8), treatment]).T
    W2 = np.vstack([np.ones(2), np.array([0,1])]).T

    dats = [dataset(W, X(7), ys[0]),
            dataset(W, X(7), ys[1]),
            dataset(W2, X(2), ys[2])]

    dist = c._wasserstein_differences(dats)
    assert np.isclose(dist, 8.333, 1e-3)

def test_tau_variances_same():
    ys = [np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
          np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
          np.array([10.,10.,10.,10.,30.,30.,30.,30.])]

    treatment = np.array([0,0,0,0,1,1,1,1,])
    W = np.vstack([np.ones(8), treatment]).T

    dats = [dataset(W, X(8), ys[i]) for i in range(3)]
    dist = c._tau_variances(dats)
    assert np.isclose(dist, 0.0, 1e-6)


def test_tau_variances_different():
    ys = [np.array([10.,10.,10.,10.,30.,30.,30.,30.]),
          np.array([20.,20.,20.,20.,30.,30.,30.,30.]),
          np.array([30.,30.,30.,30.,30.,30.,30.,30.])]

    treatment = np.array([0,0,0,0,1,1,1,1,])
    W = np.vstack([np.ones(8), treatment]).T

    dats = [dataset(W, X(8), ys[i]) for i in range(3)]
    dist = c._tau_variances(dats)
    assert np.isclose(dist, 66.666, 1e-3)

def test_tau_variances_weighted_small():
    ys = [np.array([20.,20.,20.,20.,20.,20.,30.,30.,30.,30.,30.,30.]),
          np.array([20.,20.,20.,20.,20.,20.,30.,30.,30.,30.,30.,30.]),
          np.array([40., 60.])
    ]

    treatment = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
    W = np.vstack([np.ones(12), treatment]).T
    W2 = np.vstack([np.ones(2), np.array([0,1])]).T

    dats = [dataset(W, X(12), ys[0]),
            dataset(W, X(12), ys[1]),
            dataset(W2, X(2), ys[2])]

    dist = c._tau_variances(dats)
    assert np.isclose(dist, 7.101, 1e-3)
