from .mechanisms import ReleaseMechanism


class ComposedRelease:
    """
    A base class for calculating the privacy loss of the composition
    of multiple ReleaseMechanism.
    """

    def __init__(self, mechanisms):
        # set the privacy losses to be the epsilon for each mechanism to calculate the upper limit of privacy loss
        self._privacy_losses = dict((hash(m), m.epsilon) for m in mechanisms)
        self.epsilon = self.privacy_consumed
        # set the privacy losses to the true values
        self._privacy_losses = dict((hash(m), m.privacy_consumed) for m in mechanisms)

        # set the mechanisms to report to the ComposedRelease class
        for mechanism in mechanisms:
            mechanism.accountant = self

        self.accountant = None

    def update(self, mechanism):
        self._privacy_losses[hash(mechanism)] = mechanism.privacy_consumed
        if self.accountant is not None:
            self.accountant.update(self)

    @property
    def privacy_consumed(self):
        raise NotImplementedError


class ParallelRelease(ComposedRelease):
    """
    A class for calculating the privacy loss of the composition
    of multiple ReleaseMechanisms operating over disjoint subsets of the database.

    Usage:
        accountant = PrivacyAccountant(privacy_budget=2)
        mechanisms = [LaplaceMechanism(1, 1, precision=20) for _ in range(1000)]
        parallel_mechanisms = ParallelRelease(mechanisms)
        # data releases still occur with the mechanisms
        mechanisms[0].release(data)

    """

    @property
    def privacy_consumed(self):
        return max(p for _, p in self._privacy_losses.items())


class SequentialRelease(ComposedRelease):
    """
    A class for calculating the privacy loss of the composition
    of multiple ReleaseMechanisms.

    Usage:
        accountant = PrivacyAccountant(privacy_budget=2)
        mechanisms = [LaplaceMechanism(1, 1, precision=20) for _ in range(1000)]
        parallel_mechanisms = ParallelRelease(mechanisms)
        # data releases still occur with the mechanisms
        mechanisms[0].release(data)

    """

    @property
    def privacy_consumed(self):
        return sum(p for _, p in self._privacy_losses.items())
