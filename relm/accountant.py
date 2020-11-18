class PrivacyAccountant:
    """
    This class tracks the privacy consumption of multiple mechanisms.

    Args:
        privacy_budget: the maximum privacy loss allowed across all mechanisms
    """

    def __init__(self, privacy_budget):
        self.privacy_budget = privacy_budget
        self._privacy_losses = dict()
        self._max_privacy_loss = 0

    @property
    def privacy_consumed(self):
        return sum(p for _, p in self._privacy_losses.items())

    def update(self, mechanism):
        self._privacy_losses[hash(mechanism)] = mechanism.privacy_consumed

    def add_mechanism(self, mechanism):
        """
        Adds a mechanism to be tracked by the privacy accountant.

        Args:
            mechanism: a ReleaseMechanism to be tracked

        """

        if mechanism.accountant is not None:
            raise RuntimeError(
                "mechanism: attempted to add a mechanism to two accountants."
            )

        if self._max_privacy_loss + mechanism.epsilon > self.privacy_budget:
            raise ValueError(
                f"mechanism: using this mechanism could exceed the privacy budget of {self.privacy_budget}"
                f" with a total privacy loss of {self._max_privacy_loss + mechanism.epsilon}"

            )
        mechanism.accountant = self
        self._privacy_losses[hash(mechanism)] = mechanism.privacy_consumed
        self._max_privacy_loss += mechanism.epsilon
