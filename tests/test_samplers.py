from differential_privacy.samplers import (
    uniform,
    exponential,
    laplace,
    geometric,
    two_sided_geometric,
    uniform_double,
)
import numpy as np


def _test_distribution(benchmark, func, mean, var, name=None):

    for num in [1, 20, 100, 1000]:
        samples = func(num)
        assert samples.shape == (num,)

    large_sample = func(10000000)
    assert np.isclose(large_sample.mean(), mean, rtol=0.01, atol=0.01)
    assert np.isclose(large_sample.var(), var, rtol=0.01, atol=0.01)
    if name is not None:
        pass
    benchmark(lambda: func(1000000))


def test_uniform(benchmark):
    func = uniform
    _test_distribution(benchmark, func, 0.5, 1 / 12)


def test_exponential(benchmark):
    scale = np.random.random() * 10
    mean = scale
    var = scale ** 2
    func = lambda n: exponential(n, scale)
    _test_distribution(benchmark, func, mean, var)


def test_laplace(benchmark):
    scale = np.random.random() * 10
    mean = 0
    var = 2 * scale ** 2
    func = lambda n: laplace(n, scale)
    _test_distribution(benchmark, func, mean, var)


def test_geometric(benchmark):
    scale = np.random.random()
    mean = (1 - scale) / scale
    var = (1 - scale) / scale ** 2
    func = lambda n: geometric(n, scale)
    _test_distribution(benchmark, func, mean, var)


def test_two_sided_geometric(benchmark):
    scale = np.random.random()
    mean = 0
    var = 2 * scale / (1 - scale) ** 2
    func = lambda n: two_sided_geometric(n, scale)
    _test_distribution(benchmark, func, mean, var)


def test_uniform_double(benchmark):
    func = uniform_double
    _test_distribution(benchmark, func, 0.5, 1 / 12)
