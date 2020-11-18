class PrivacyAccountant:
    """
    This class tracks the privacy consumption of multiple mechanisms.

    Args:
        privacy_budget: the maximum privacy loss allowed across all mechanisms
    """

    def __init__(self, privacy_budget):
        self.privacy_budget = privacy_budget
        self._privacy_losses = dict()
        self._standard_mechanisms = set()
        self._disjoint_mechanism_groups = []
        self._max_privacy_loss = 0

    @property
    def privacy_consumed(self):
        _privacy_consumed = sum(p for _, p in self._privacy_losses.items())
        for disjoint_mechanisms in self._disjoint_mechanism_groups:
            _privacy_consumed += max(self._privacy_losses[m] for m in disjoint_mechanisms)
            _privacy_consumed -= sum(self._privacy_losses[m] for m in disjoint_mechanisms)

        return _privacy_consumed

    def update(self, mechanism):
        self._privacy_losses[hash(mechanism)] = mechanism.privacy_consumed

    def add_mechanism(self, mechanism):
        """
        Adds a mechanism to be tracked by the privacy accountant.

        Args:
            mechanism: a ReleaseMechanism to be tracked

        """

        if self._max_privacy_loss + mechanism.epsilon > self.privacy_budget:
            raise ValueError(
                f"mechanism: using this mechanism could exceed the privacy budget of {self.privacy_budget}"
                f" with a total privacy loss of {self._max_privacy_loss + mechanism.epsilon}"
            )
        self._add_mechanism(mechanism)
        self._max_privacy_loss += mechanism.epsilon

    def add_disjoint_mechanisms(self, mechanisms):

        max_epsilon = max(mechanism.epsilon for mechanism in mechanisms)
        if self._max_privacy_loss + max_epsilon > self.privacy_budget:
            raise ValueError(
                f"mechanism: using this mechanism could exceed the privacy budget of {self.privacy_budget}"
                f" with a total privacy loss of {self._max_privacy_loss + max_epsilon}"
            )

        for mechanism in mechanisms:
            self._add_mechanism(mechanism)

        self._disjoint_mechanism_groups.append(set(hash(mechanism) for mechanism in mechanisms))
        self._max_privacy_loss += max_epsilon

    def _add_mechanism(self, mechanism):
        if mechanism.accountant is not None:
            raise RuntimeError(
                "mechanism: attempted to add a mechanism to two accountants."
            )

        mechanism.accountant = self
        self._privacy_losses[hash(mechanism)] = mechanism.privacy_consumed
