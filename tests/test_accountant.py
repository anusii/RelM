import numpy as np
from relm.mechanisms import LaplaceMechanism
from relm.accountant import PrivacyAccountant


def test_add_mechanism():
    accountant = PrivacyAccountant(10)
    n = np.random.randint(3, 100)

    for _ in range(n):
        mechanism = LaplaceMechanism(1, precision=20, sensitivity=1)
        accountant.add_mechanism(mechanism)

    assert len(accountant.mechanisms) == n


def test_privacy_consumed():
    accountant = PrivacyAccountant(1000000)
    epsilons = 10 * np.random.random(10)

    vals = np.zeros(10)
    for i, e in enumerate(epsilons):
        mechanism = LaplaceMechanism(e, precision=20, sensitivity=1)
        accountant.add_mechanism(mechanism)
        _ = mechanism.release(vals)
        assert accountant.privacy_consumed == epsilons[: i + 1].sum()


def test_check_valid():
    accountant = PrivacyAccountant(20)
    vals = np.zeros(10)
    for _ in range(20):
        assert accountant.check_valid()
        mechanism = LaplaceMechanism(1, precision=20, sensitivity=1)
        accountant.add_mechanism(mechanism)
        _ = mechanism.release(vals)

    assert not accountant.check_valid()
