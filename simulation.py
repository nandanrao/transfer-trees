from copy import deepcopy
from collections import namedtuple
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

abcs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

def plot_dat(vars_):
    K = len(vars_)
    _, axes = plt.subplots(K, 1, sharex=True, figsize=(20, 10))

    for (H, title), ax in zip(vars_, axes):
        for i, h in zip(abcs[:K], H):
            sns.distplot(h, label=i, ax=ax)
        ax.legend()
        ax.set_title(title)


def generate_one(N, fn, hidden_cause, hidden, v_cond, z_cond):
    K = len(hiddens)
    # H is latent variable, distribution changes (not )
    H = [np.random.normal(loc=a, scale=b, size=N) for a, b in hiddens]

    # V := f(H, N_X)
    # V = [c*h + np.random.normal(loc=a, scale=b, size=N) for h, (c, a, b) in zip(H, v_conds)]
    V = [c*h + np.random.uniform(low=a, high=a+b, size=N) for h, (c, a, b) in zip(H, v_conds)]

    # if hidden_cause:
    # H -> V, H -> Y
    # else:
    # V -> H, H -> Y
    if not hidden_cause:
        V, H = deepcopy(H), deepcopy(V)

    # Z = [gamma.rvs(int(np.random.normal(40, 10)), loc=0, scale=1, size=N) for h in H]
    # Z := f(N_Z)
    Z = [np.random.normal(loc=a, scale=b, size=N) for _, (a, b) in zip(H, z_conds)]

    # W := f(N_W) -- TREATMENT
    W = [np.random.binomial(1, 0.5, size=N) for h in H]

    # Y:= fn(H, V, Z, W, N_Y)
    Y = [fn(h,v,z,w) for h,v,z,w in zip(H, V, Z, W)]

    taus = [fn(h, v, z, 1) - fn(h, v, z, 0) for h, v, z in zip(H, V, Z)]

    if plot:
        plot_dat([(H, 'H'), (V, 'V'), (Z, 'Z'), (Y, 'Y'), (taus, 'Tau')])

    return [(y, np.array([w, v, z]).T, tau) for y, v, z, w, h, tau in zip(Y, V, Z, W, H, taus)]   


def generate_data(N, fn, hidden_cause, plot, hiddens, v_conds, z_conds):
    K = len(hiddens)
    # H is latent variable, distribution changes (not )
    H = [np.random.normal(loc=a, scale=b, size=N) for a, b in hiddens]

    # V := f(H, N_X)
    V = [c*h + np.random.normal(loc=a, scale=b, size=N) for h, (c, a, b) in zip(H, v_conds)]

    # if hidden_cause:
    # H -> V, H -> Y
    # else:
    # V -> H, H -> Y
    if not hidden_cause:
        V, H = deepcopy(H), deepcopy(V)

    # Z = [gamma.rvs(int(np.random.normal(40, 10)), loc=0, scale=1, size=N) for h in H]
    # Z := f(N_Z)
    Z = [np.random.normal(loc=a, scale=b, size=N) for _, (a, b) in zip(H, z_conds)]

    # W := f(N_W) -- TREATMENT
    W = [np.random.binomial(1, 0.5, size=N) for h in H]

    fn_res = [fn(h, v, z, 1) for h, v, z in zip(H, V, Z)]
    taus = [t for t, _ in fn_res]

    # Y:= fn(H, V, Z, W, N_Y)
    Y = [t*w + noise for (t, noise), w in zip(fn_res, W)]


    if plot:
        plot_dat([(H, 'H'), (V, 'V'), (Z, 'Z'), (Y, 'Y'), (taus, 'Tau')])

    return [(y, np.array([w, v, z]).T, tau) for y, v, z, w, h, tau in zip(Y, V, Z, W, H, taus)]


def flatten(a):
    return np.array([y for x in a for y in x])



Ctx = namedtuple('Ctx', ['source', 'target'])
Dataset = namedtuple('Dataset', ['phi', 'y', 'treatment', 'context_idxs', 'tau'])

def split_out_dat(dat):
    phi = PolynomialFeatures(degree=1, include_bias=False).fit_transform
    ys, Xs, taus = zip(*dat)

    ys_source, Xs_source = np.concatenate(ys[:-1]), np.concatenate(Xs[:-1])
    ys_target, Xs_target = ys[-1], Xs[-1]

    phi_source = phi(Xs_source[:, 1:])
    phi_target = phi(Xs_target[:, 1:])

    treatment = Xs_source[:, 0]
    N = ys[0].shape[0]

    context_idxs = np.array([j for i, _ in enumerate(ys[:-1]) for j in [i]*N])

    idx = np.arange(phi_source.shape[0])
    np.random.shuffle(idx)

    phi_source = phi_source[idx, :]
    ys_source = ys_source[idx]
    treatment = treatment[idx]
    context_idxs = context_idxs[idx]
    taus_source, taus_target = flatten(taus[:-1])[idx], taus[-1]

    return Dataset(Ctx(phi_source, phi_target),
                   Ctx(ys_source, ys_target),
                   treatment,
                   context_idxs,
                   Ctx(taus_source, taus_target))
