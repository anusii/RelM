from differential_privacy.samplers import uniform, exponential, laplace, geometric, two_sided_geometric
import numpy as np


def test_uniform():
    for num in [1, 20, 100, 1000]:
        samples = uniform(num)
        assert samples.shape == (num,)

    large_sample = uniform(10000000)
    assert np.isclose(large_sample.mean(), 0.5, rtol=0.01, atol=0.01)
    assert np.isclose(large_sample.var(), 1 / 12, rtol=0.01, atol=0.01)


def test_exponential():
    for num in [1, 20, 100, 1000]:
        samples = exponential(num, 1.0)
        assert samples.shape == (num,)

    scale = np.random.random() * 10
    large_sample = exponential(10000000, scale)
    assert np.isclose(large_sample.mean(), scale, rtol=0.01, atol=0.01)
    assert np.isclose(large_sample.var(), scale ** 2, rtol=0.01, atol=0.01)


def test_laplace():
    for num in [1, 20, 100, 1000]:
        samples = laplace(num, 1.0)
        assert samples.shape == (num,)

    scale = np.random.random() * 10
    large_sample = laplace(10000000, scale)
    assert np.isclose(large_sample.mean(), 0, rtol=0.01, atol=0.01)
    assert np.isclose(large_sample.var(), 2 * scale ** 2, rtol=0.01, atol=0.01)


# @pytest.mark.xfail(reason="possible bug in implementation")
def test_geometric():
    for num in [1, 20, 100, 1000]:
        samples = geometric(num, 0.5)
        assert samples.shape == (num,)

    scale = np.random.random()
    large_sample = geometric(10000000, scale)
    assert np.isclose(large_sample.mean(), (1 - scale) / scale, rtol=0.01, atol=0.01)
    assert np.isclose(large_sample.var(), (1 - scale) / scale ** 2, rtol=0.01, atol=0.01)


def test_two_sided_geometric():
    for num in [1, 20, 100, 1000]:
        samples = two_sided_geometric(num, 0.5)
        assert samples.shape == (num,)

    scale = 0.5
    large_sample = two_sided_geometric(10000000, scale)
    assert np.isclose(large_sample.mean(), 0, rtol=0.01, atol=0.01)
    assert np.isclose(large_sample.var(), 2 * (scale) / (1 - scale) ** 2, rtol=0.01, atol=0.01)
