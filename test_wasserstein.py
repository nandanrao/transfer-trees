import numpy as np
from wasserstein import *
import scipy.stats

def test_cdf():
    a = np.random.normal(size=20)
    b = np.random.normal(size=20)
    res = wasserstein_distance(a, b)
    assert np.isclose(res, scipy.stats.wasserstein_distance(a, b))
