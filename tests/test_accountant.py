import numpy as np
from relm.mechanisms import LaplaceMechanism
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
        assert np.isclose(accountant.privacy_consumed, epsilons[: i + 1].sum())


def test_add_disjoint_mechanisms():

    accountant = PrivacyAccountant(1000000)

    mechanisms = [
        LaplaceMechanism(20, precision=20, sensitivity=1) for _ in range(1000)
    ]
    accountant.add_disjoint_mechanisms(mechanisms)

    mechanisms = [LaplaceMechanism(4, precision=20, sensitivity=1) for _ in range(1000)]
    accountant.add_disjoint_mechanisms(mechanisms)

    accountant.add_mechanism(LaplaceMechanism(100, precision=20, sensitivity=1))

    values = np.zeros(1)
    _ = [mechanism.release(values) for mechanism in mechanisms]
    assert accountant.privacy_consumed == 4
    assert accountant._max_privacy_loss == 124
