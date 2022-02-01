import numpy as np
from relm.mechanisms import LaplaceMechanism, GaussianMechanism
from relm.accountant import PrivacyAccountant
import pytest


def test_add_mechanism():
    accountant = PrivacyAccountant(10)

    for _ in range(10):
        mechanism = LaplaceMechanism(1, precision=20, sensitivity=1)
        accountant.add_mechanism(mechanism)

    assert len(accountant._privacy_losses) == 10

    with pytest.raises(ValueError):
        mechanism = LaplaceMechanism(1, precision=20, sensitivity=1)
        accountant.add_mechanism(mechanism)


def test_privacy_consumed():
    accountant = PrivacyAccountant(1000000)
    epsilons = 10 * np.random.random(10)

    vals = np.zeros(10)
    for i, e in enumerate(epsilons):
        mechanism = LaplaceMechanism(e, precision=20, sensitivity=1)
        accountant.add_mechanism(mechanism)
        _ = mechanism.release(vals)
        assert np.isclose(accountant.privacy_consumed[0], epsilons[: i + 1].sum())
        assert np.isclose(accountant.privacy_consumed[1], 0)

    accountant = PrivacyAccountant(epsilon_budget=1000000, delta_budget=1000000)
    epsilons = np.random.random(10)
    deltas = np.random.random(10)

    vals = np.zeros(10)
    for i, (e, d) in enumerate(zip(epsilons, deltas)):
        mechanism = GaussianMechanism(epsilon=e, delta=d, precision=20, sensitivity=1)
        accountant.add_mechanism(mechanism)
        _ = mechanism.release(vals)
        assert np.isclose(accountant.privacy_consumed[0], epsilons[: i + 1].sum())
        assert np.isclose(accountant.privacy_consumed[1], deltas[: i + 1].sum())
