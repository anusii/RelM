from relm.mechanisms import LaplaceMechanism
from relm.accountant import PrivacyAccountant
from relm.composition import ParallelRelease, SequentialRelease
import numpy as np


def test_parallel_release():
    accountant = PrivacyAccountant(1000000)

    mechanisms = [
        LaplaceMechanism(20, precision=20, sensitivity=1) for _ in range(1000)
    ]
    para_mechs = ParallelRelease(mechanisms)
    accountant.add_mechanism(para_mechs)

    mechanisms = [LaplaceMechanism(4, precision=20, sensitivity=1) for _ in range(1000)]
    para_mechs = ParallelRelease(mechanisms)
    accountant.add_mechanism(para_mechs)

    accountant.add_mechanism(LaplaceMechanism(100, precision=20, sensitivity=1))

    values = np.zeros(1)
    _ = [mechanism.release(values) for mechanism in mechanisms]

    assert accountant.privacy_consumed == (4, 0)
    assert accountant.epsilon_allocated == 124
    assert accountant.delta_allocated == 0
    assert para_mechs.privacy_consumed == (4, 0)
    assert para_mechs.epsilon == 4


def test_sequential_release():
    accountant = PrivacyAccountant(1000000)

    mechanisms = [
        LaplaceMechanism(20, precision=20, sensitivity=1) for _ in range(1000)
    ]
    seq_mechs = SequentialRelease(mechanisms)
    accountant.add_mechanism(seq_mechs)

    values = np.zeros(1)
    _ = [mechanism.release(values) for mechanism in mechanisms]

    assert accountant.privacy_consumed == (20000, 0)
    assert seq_mechs.privacy_consumed == (20000, 0)
    assert seq_mechs.epsilon == 20000
