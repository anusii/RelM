from differential_privacy.samplers import (
    uniform,
    exponential,
    laplace,
    geometric,
    two_sided_geometric,
    uniform_double,
)
from differential_privacy import backend
import numpy as np
from crlibm import log_rn
import scipy.stats


def _test_distribution(benchmark, func, mean, var, control=None):

    for num in [4, 20, 100, 1000]:
        samples = func(num)
        assert samples.shape == (num,)

    large_sample = func(10000000)
    print(large_sample[:10])
    sample_mean, sample_var = large_sample.mean(), large_sample.var()
    benchmark(lambda: func(1000000))
    assert np.isclose(sample_mean, mean, rtol=0.01, atol=0.01)
    assert np.isclose(sample_var, var, rtol=0.01, atol=0.01)
    if control is not None:
        large_control = control(10000000)
        score, pval = scipy.stats.ks_2samp(large_sample, large_control)
        assert pval > 0.001


def test_uniform(benchmark):
    func = uniform
    control = lambda n: scipy.stats.uniform.rvs(size=n)
    _test_distribution(benchmark, func, 0.5, 1 / 12, control)


def test_exponential(benchmark):
    scale = np.random.random() * 10
    mean = scale
    var = scale ** 2
    func = lambda n: exponential(n, scale)
    control = lambda n: scipy.stats.expon.rvs(loc=0, scale=scale, size=n)
    _test_distribution(benchmark, func, mean, var, control)


def test_laplace(benchmark):
    scale = np.random.random() * 10
    mean = 0
    var = 2 * scale ** 2
    func = lambda n: laplace(n, scale)
    control = lambda n: scipy.stats.laplace.rvs(scale=scale, size=n)
    _test_distribution(benchmark, func, mean, var, control)


def test_geometric(benchmark):
    scale = np.random.random()
    mean = (1 - scale) / scale
    var = (1 - scale) / scale ** 2
    func = lambda n: geometric(n, scale)
    control = lambda n: scipy.stats.geom.rvs(p=scale, size=n) - 1
    _test_distribution(benchmark, func, mean, var, control)


def test_two_sided_geometric(benchmark):
    scale = np.random.random()
    mean = 0
    var = 2 * scale / (1 - scale) ** 2
    func = lambda n: two_sided_geometric(n, scale)
    f = lambda n: scipy.stats.geom.rvs(p=1 - scale, size=n)
    g = lambda n: scipy.stats.geom.rvs(p=1 - scale, size=n)
    control = lambda n: f(n) - g(n)
    _test_distribution(benchmark, func, mean, var, control)


def test_uniform_double(benchmark):
    func = uniform_double
    control = lambda n: scipy.stats.uniform.rvs(size=n)
    _test_distribution(benchmark, func, 0.5, 1 / 12, control)


def test_ln_rn():
    for _ in range(100000):
        x = np.random.random() / np.random.random()
        assert backend.ln_rn(x) == log_rn(x)


def test_coin_flip(benchmark):
    bias = np.random.random()
    func = lambda: np.mean([backend.coin_flip(bias) for _ in range(1000000)])
    mean = func()
    assert np.isclose(mean, bias, rtol=0.01)
    benchmark(func)


def test_laplace_fixed_point(benchmark):
    scale = np.random.random() * 10
    mean = 0
    var = 2 * scale ** 2
    func = lambda n: backend.fixed_point_laplace(scale, n)
    control = None  # lambda n: scipy.stats.laplace.rvs(scale=scale, size=n)
    _test_distribution(benchmark, func, mean, var, control)
