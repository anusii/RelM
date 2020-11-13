# from differential_privacy.samplers import (
#     fixed_point_laplace,
#     fixed_point_exponential,
#     geometric,
#     uniform_double,
# )
from differential_privacy import backend
import numpy as np
from crlibm import log_rn
import scipy.stats


def _test_distribution(benchmark, func, mean, var, control=None):

    for num in [1, 20, 100, 1000]:
        samples = func(num)
        assert samples.shape == (num,)

    large_sample = func(10000000)
    assert np.isclose(large_sample.mean(), mean, rtol=0.01, atol=0.01)
    assert np.isclose(large_sample.var(), var, rtol=0.01, atol=0.01)
    if control is not None:
        large_control = control(10000000)
        score, pval = scipy.stats.ks_2samp(large_sample, large_control)
        assert pval > 0.001
    benchmark(lambda: func(1000000))


# def test_fixed_point_laplace(benchmark):
#     scale = np.random.random() * 10
#     mean = 0
#     var = 2 * scale ** 2
#     func = lambda n: fixed_point_laplace(n, scale, 35) * 2.0 ** (-35)
#     control = lambda n: scipy.stats.laplace.rvs(scale=scale, size=n)
#     _test_distribution(benchmark, func, mean, var, control)


# def test_fixed_point_exponential(benchmark):
#     scale = np.random.random() * 10
#     mean = scale
#     var = scale ** 2
#     func = lambda n: fixed_point_exponential(n, scale, 35) * 2.0 ** (-35)
#     control = lambda n: scipy.stats.expon.rvs(scale=scale, size=n)
#     _test_distribution(benchmark, func, mean, var, control)
#
#
# def test_geometric(benchmark):
#     p = np.random.random()
#     mean = (1 - p) / p
#     var = (1 - p) / p ** 2
#     func = lambda n: geometric(n, p)
#     control = lambda n: scipy.stats.geom.rvs(p=p, size=n) - 1
#     _test_distribution(benchmark, func, mean, var, control)
#
#
# def test_uniform_double(benchmark):
#     func = uniform_double
#     control = lambda n: scipy.stats.uniform.rvs(size=n)
#     _test_distribution(benchmark, func, 0.5, 1 / 12, control)
#
#
# def test_ln_rn():
#     for _ in range(100000):
#         x = np.random.random() / np.random.random()
#         assert backend.ln_rn(x) == log_rn(x)
