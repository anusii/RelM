import numpy as np


class ReleaseMechanism:
    def __init__(self, epsilon, delta=0):
        self.epsilon = epsilon
        self.delta = delta
        self._is_valid = True
        self.accountant = None
        self._id = np.random.randint(low=0, high=2 ** 60)

    def _check_valid(self):

        if not self._is_valid:
            raise RuntimeError(
                "Mechanism has exhausted has exhausted its privacy budget."
            )

    def _update_accountant(self):
        if self.accountant is not None:
            self.accountant.update(self)

    def __hash__(self):
        return self._id

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy array of the output of a query.

        Returns:
            A numpy array of perturbed values.
        """
        raise NotImplementedError()

    @property
    def privacy_consumed(self):
        """
        Computes the privacy budget consumed by the mechanism so far.
        """
        if self._is_valid:
            return [0, 0]
        else:
            return [self.epsilon, self.delta]
