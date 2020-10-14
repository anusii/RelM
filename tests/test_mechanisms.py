import numpy as np
import pytest
from differential_privacy.mechanisms import (
    LaplaceMechanism,
    GeometricMechanism,
    Snapping,
    AboveThreshold,
    SparseIndicator,
    SparseNumeric,
)


def _test_mechanism(benchmark, mechanism):
    data = np.random.random(100000)
    benchmark.pedantic(lambda: mechanism.release(data), iterations=1, rounds=1)
    with pytest.raises(RuntimeError):
        mechanism.release(data)


def test_laplace(benchmark):
    mechanism = LaplaceMechanism(epsilon=1)
    _test_mechanism(benchmark, mechanism)


def test_geometric(benchmark):
    mechanism = GeometricMechanism(epsilon=1)
    _test_mechanism(benchmark, mechanism)


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


def test_snapping(benchmark):
    mechanism = Snapping(epsilon=1.0, B=10)
    _test_mechanism(benchmark, mechanism)
