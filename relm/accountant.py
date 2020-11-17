class PrivacyAccountant:
    """
    This class tracks the privacy consumption of multiple mechanisms.

    Args:
        privacy_budget: the maximum privacy loss allowed across all mechanisms
    """

    def __init__(self, privacy_budget):
        self.privacy_budget = privacy_budget
        self._privacy_losses = dict()

    @property
    def privacy_consumed(self):
        return sum(p for _, p in self._privacy_losses.items())

    def check_valid(self):
        """
        Checks that the privacy consumed is less than the privacy budget.

        Returns:
            bool
        """
        is_valid = self.privacy_consumed < self.privacy_budget
        return is_valid

    def update(self, mechanism):
        self._privacy_losses[hash(mechanism)] = mechanism.get_privacy_consumption()

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
        mechanism.accountant = self
        self._privacy_losses[hash(mechanism)] = mechanism.get_privacy_consumption()
