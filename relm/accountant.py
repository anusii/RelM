class PrivacyAccountant:
    """
    This class tracks the privacy consumption of multiple mechanisms.

    Args:
        privacy_budget: the maximum privacy loss allowed across all mechanisms
    """

    def __init__(self, privacy_budget):
        self.privacy_budget = privacy_budget
        self.mechanisms = []

    @property
    def privacy_consumed(self):
        return sum(m.get_privacy_consumption() for m in self.mechanisms)

    def check_valid(self):
        """
        Checks that the privacy consumed is less than the privacy budget.

        Returns:
            bool
        """
        is_valid = self.privacy_consumed < self.privacy_budget
        return is_valid

    def add_mechanism(self, mechanism):
        """
        Adds a mechanism to be tracked by the privacy accountant.

        Args:
            mechanism: a ReleaseMechanism to be tracked

        """

        # connect the account to the mechanisms
        # creates a circular loop
        mechanism.accountant = self
        self.mechanisms.append(mechanism)
