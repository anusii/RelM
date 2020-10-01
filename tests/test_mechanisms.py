import numpy as np
import pytest
from differential_privacy.mechanisms import (
    LaplaceMechanism,
    GeometricMechanism,
    Snapping,
    Sparse,
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


def test_sparse(benchmark):
    mechanism = Sparse(epsilon=1, threshold=0.1, cutoff=100)
    _test_mechanism(benchmark, mechanism)
    mechanism = Sparse(epsilon=1, threshold=0.01, cutoff=100)
    data = np.random.random(1000)
    assert len(mechanism.release(data) == 100)


def test_sparse_numeric(benchmark):
    mechanism = SparseNumeric(epsilon=1, threshold=0.1, cutoff=100)
    _test_mechanism(benchmark, mechanism)
    mechanism = Sparse(epsilon=1, threshold=0.01, cutoff=100)
    data = np.random.random(1000)
    assert len(mechanism.release(data) == 100)
