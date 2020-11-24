import numpy as np
import scipy.stats
import pytest
from relm.mechanisms import (
    LaplaceMechanism,
    GeometricMechanism,
    ExponentialMechanism,
    SnappingMechanism,
    AboveThreshold,
    SparseIndicator,
    SparseNumeric,
    ReportNoisyMax,
)


def _test_mechanism(benchmark, mechanism, dtype=np.float64):
    data = np.random.random(100000).astype(dtype)
    benchmark.pedantic(lambda: mechanism.release(data), iterations=1, rounds=1)
    with pytest.raises(RuntimeError):
        mechanism.release(data)


def test_LaplaceMechanism(benchmark):
    mechanism = LaplaceMechanism(epsilon=1, sensitivity=1, precision=35)
    _test_mechanism(benchmark, mechanism)
    # Goodness of fit test
    mechanism = LaplaceMechanism(epsilon=1, sensitivity=1, precision=35)
    data = np.random.random(10000000) * 100
    values = mechanism.release(data)
    control = scipy.stats.laplace.rvs(scale=1.0 + 2 ** -35, size=data.size)
    score, pval = scipy.stats.ks_2samp(values - data, control)
    assert pval > 0.001


def test_GeometricMechanism(benchmark):
    mechanism = GeometricMechanism(epsilon=1, sensitivity=1)
    _test_mechanism(benchmark, mechanism, dtype=np.int64)
    # Goodness of fit test
    epsilon = 0.01
    mechanism = GeometricMechanism(epsilon=epsilon, sensitivity=1)
    n = 10000000
    data = np.random.randint(0, 2 ** 16, size=n, dtype=np.int64)
    values = mechanism.release(data)
    q = np.exp(-epsilon)
    x = scipy.stats.geom.rvs(p=1 - q, size=n)
    y = scipy.stats.geom.rvs(p=1 - q, size=n)
    z = x - y
    score, pval = scipy.stats.ks_2samp(values - data, z)
    assert pval > 0.001


def test_ExponentialMechanism(benchmark):
    n = 10
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    mechanism = ExponentialMechanism(
        epsilon=1.0,
        utility_function=utility_function,
        sensitivity=1.0,
        output_range=output_range,
    )
    _test_mechanism(benchmark, mechanism)
    # Goodness of fit test
    n = 16
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    data = np.zeros(1)
    TRIALS = 2 ** 10
    mechanism = ExponentialMechanism(
        epsilon=1.0,
        utility_function=utility_function,
        sensitivity=1.0,
        output_range=output_range,
    )
    values = mechanism.release(data, _k=TRIALS)
    z = scipy.stats.laplace.rvs(scale=2.0, size=TRIALS)
    score, pval = scipy.stats.ks_2samp(values, z)
    assert pval > 0.001


def test_above_threshold(benchmark):
    mechanism = AboveThreshold(epsilon=1, sensitivity=1.0, threshold=0.1)
    _test_mechanism(benchmark, mechanism)
    mechanism = AboveThreshold(epsilon=1, sensitivity=1.0, threshold=0.01)
    data = np.random.random(1000)
    index = mechanism.release(data)
    assert type(index) == int


def test_sparse_indicator(benchmark):
    mechanism = SparseIndicator(epsilon=1, sensitivity=1.0, threshold=0.1, cutoff=100)
    _test_mechanism(benchmark, mechanism)
    mechanism = SparseIndicator(epsilon=1, sensitivity=1.0, threshold=0.01, cutoff=100)
    data = np.random.random(1000)
    indices = mechanism.release(data)
    assert len(indices) == 100


def test_sparse_numeric(benchmark):
    mechanism = SparseNumeric(epsilon=1, sensitivity=1.0, threshold=0.1, cutoff=100)
    _test_mechanism(benchmark, mechanism)
    mechanism = SparseNumeric(epsilon=1, sensitivity=1.0, threshold=0.01, cutoff=100)
    data = np.random.random(1000)
    indices, values = mechanism.release(data)
    assert len(indices) == 100
    assert len(values) == 100


def test_SnappingMechanism(benchmark):
    mechanism = SnappingMechanism(epsilon=1.0, B=10)
    _test_mechanism(benchmark, mechanism)


def test_ReportNoisyMax(benchmark):
    mechanism = ReportNoisyMax(epsilon=0.1, precision=35)
    _test_mechanism(benchmark, mechanism)
