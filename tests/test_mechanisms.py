import numpy as np
import scipy.stats
import pytest
from relm.mechanisms import (
    LaplaceMechanism,
    GeometricMechanism,
    ExponentialMechanism,
    PermuteAndFlipMechanism,
    SnappingMechanism,
    AboveThreshold,
    SparseIndicator,
    SparseNumeric,
    ReportNoisyMax,
    MultiplicativeWeights,
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


def test_ExponentialMechanismWeightedIndex(benchmark):
    n = 8
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    mechanism = ExponentialMechanism(
        epsilon=1.0,
        utility_function=utility_function,
        sensitivity=1.0,
        output_range=output_range,
        method="weighted_index",
    )
    _test_mechanism(benchmark, mechanism)
    # Goodness of fit test
    n = 6
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    data = np.zeros(1)
    TRIALS = 2 ** 12
    values = np.empty(TRIALS)
    for i in range(TRIALS):
        mechanism = ExponentialMechanism(
            epsilon=1.0,
            utility_function=utility_function,
            sensitivity=1.0,
            output_range=output_range,
            method="weighted_index",
        )
        values[i] = mechanism.release(data)

    z = scipy.stats.laplace.rvs(scale=2.0, size=TRIALS)
    score, pval = scipy.stats.ks_2samp(values, z)
    assert pval > 0.001


def test_ExponentialMechanismGumbelTrick(benchmark):
    n = 8
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    mechanism = ExponentialMechanism(
        epsilon=1.0,
        utility_function=utility_function,
        sensitivity=1.0,
        output_range=output_range,
        method="gumbel_trick",
    )
    _test_mechanism(benchmark, mechanism)
    # Goodness of fit test
    n = 6
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    data = np.zeros(1)
    TRIALS = 2 ** 12
    values = np.empty(TRIALS)
    for i in range(TRIALS):
        mechanism = ExponentialMechanism(
            epsilon=1.0,
            utility_function=utility_function,
            sensitivity=1.0,
            output_range=output_range,
            method="gumbel_trick",
        )
        values[i] = mechanism.release(data)

    z = scipy.stats.laplace.rvs(scale=2.0, size=TRIALS)
    score, pval = scipy.stats.ks_2samp(values, z)
    assert pval > 0.001


def test_ExponentialMechanismSampleAndFlip(benchmark):
    n = 8  # This gets *really* slow if n in mcuh bigger than 6.
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    mechanism = ExponentialMechanism(
        epsilon=1.0,
        utility_function=utility_function,
        sensitivity=1.0,
        output_range=output_range,
        method="sample_and_flip",
    )
    _test_mechanism(benchmark, mechanism)
    # Goodness of fit test
    n = 6
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    data = np.zeros(1)
    TRIALS = 2 ** 12
    values = np.empty(TRIALS)
    for i in range(TRIALS):
        mechanism = ExponentialMechanism(
            epsilon=1.0,
            utility_function=utility_function,
            sensitivity=1.0,
            output_range=output_range,
            method="sample_and_flip",
        )
        values[i] = mechanism.release(data)

    z = scipy.stats.laplace.rvs(scale=2.0, size=TRIALS)
    score, pval = scipy.stats.ks_2samp(values, z)
    assert pval > 0.001


def test_PermuteAndFlipMechanism(benchmark):
    n = 8  # This gets *really* slow if n in mcuh bigger than 6.
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    mechanism = PermuteAndFlipMechanism(
        epsilon=1.0,
        utility_function=utility_function,
        sensitivity=1.0,
        output_range=output_range,
    )
    _test_mechanism(benchmark, mechanism)
    # Goodness of fit test
    n = 6
    output_range = np.arange(-(2 ** (n - 1)), 2 ** (n - 1) - 1, 2 ** -10)
    utility_function = lambda x: -np.abs(output_range - np.mean(x))
    data = np.zeros(1)
    TRIALS = 2 ** 12
    values = np.empty(TRIALS)
    for i in range(TRIALS):
        mechanism = PermuteAndFlipMechanism(
            epsilon=1.0,
            utility_function=utility_function,
            sensitivity=1.0,
            output_range=output_range,
        )
        values[i] = mechanism.release(data)

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


def test_MultiplicativeWeights():
    data = np.random.randint(0, 10, 1000)
    query = np.random.randint(0, 2, 1000)
    queries = [query] * 1000

    # test privacy consumption
    mechanism = MultiplicativeWeights(50, 25, 0, 0.1, data)
    with pytest.raises(RuntimeError):
        results = mechanism.release(queries)

    assert mechanism.privacy_consumed == 50

    # ridiculous mechanism to test convergence
    mechanism = MultiplicativeWeights(10000, 2000, 100, 0.5, data)
    _ = mechanism.release([query])
    results = mechanism.release(queries)
    assert len(results) == len(queries)
    assert np.isclose(results.mean(), (query * data).sum(), rtol=0.1)
    assert (
        abs((mechanism.data_est * query).sum() * data.sum() - (data * query).sum())
        < 100
    )
