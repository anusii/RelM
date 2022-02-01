class PrivacyAccountant:
    """
    This class tracks the privacy consumption of multiple mechanisms.

    Args:
        privacy_budget: the maximum privacy loss allowed across all mechanisms
    """

    def __init__(self, epsilon_budget, delta_budget=0):
        self.epsilon_budget = epsilon_budget
        self.delta_budget = delta_budget
        self._privacy_losses = dict()
        self.epsilon_allocated = 0
        self.delta_allocated = 0

    @property
    def privacy_consumed(self):
        epsilon_consumed = sum(p[0] for _, p in self._privacy_losses.items())
        delta_consumed = sum(p[1] for _, p in self._privacy_losses.items())
        return (epsilon_consumed, delta_consumed)

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

        if self.epsilon_allocated + mechanism.epsilon > self.epsilon_budget:
            raise ValueError(
                f"mechanism: using this mechanism could exceed the privacy budget of {self.epsilon_budget}"
                f" with a total privacy alocated of {self.epsilon_allocated + mechanism.epsilon}"
            )
        if self.delta_allocated + mechanism.delta > self.delta_budget:
            raise ValueError(
                f"mechanism: using this mechanism could exceed the privacy budget of {self.delta_budget}"
                f" with a total privacy alocated of {self.delta_allocated + mechanism.delta}"
            )
        mechanism.accountant = self
        self._privacy_losses[hash(mechanism)] = mechanism.privacy_consumed
        self.epsilon_allocated += mechanism.epsilon
        self.delta_allocated += mechanism.delta
