from .base import ReleaseMechanism
from relm import backend


class ExponentialMechanism(ReleaseMechanism):
    """
    Insecure implementation of the Exponential Mechanism. This mechanism can be used once
    after which its privacy budget will be exhausted and it can no longer be used.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        utility_function: the utility function. This should accpet an array of values
                          produced by the query function and return an 1D array of
                          utilities of the same size as output_range.
        sensitivity: the sensitivity of the utility function.
        output_range: an array of possible output values for the mechanism.
        method: a string that specifies which algorithm will be used to sample
                from the output distribution. Currently, three options are supported:
                "weighted_index", "gumbel_trick", and "sample_and_flip".
    """

    def __init__(
        self,
        epsilon,
        utility_function,
        sensitivity,
        output_range,
        method="gumbel_trick",
    ):
        super(ExponentialMechanism, self).__init__(epsilon)
        self.utility_function = utility_function
        self.sensitivity = sensitivity
        self.output_range = output_range
        self.method = method

        self.effective_epsilon = self.epsilon / (2.0 * sensitivity)

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy array of the output of a query.

        Returns:
            An element of output_range set of the mechanism.
        """

        self._check_valid()
        self._is_valid = False
        self._update_accountant()

        utilities = self.utility_function(values)
        #print("########", utilities)
        index = self._sampler(utilities)
        return self.output_range[index]

    @property
    def privacy_consumed(self):
        """
        Computes the privacy budget consumed by the mechanism so far.
        """
        if self._is_valid:
            return 0
        else:
            return self.epsilon

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        if value == "weighted_index":
            sampler = lambda utilities: backend.exponential_mechanism_weighted_index(
                utilities, self.effective_epsilon
            )
        elif value == "gumbel_trick":
            sampler = lambda utilities: backend.exponential_mechanism_gumbel_trick(
                utilities, self.effective_epsilon
            )
        elif value == "sample_and_flip":
            sampler = lambda utilities: backend.exponential_mechanism_sample_and_flip(
                utilities, self.effective_epsilon
            )
        else:
            raise ValueError("Sampling method '%s' not supported." % method)

        self._method = value
        self._sampler = sampler


class PermuteAndFlipMechanism(ReleaseMechanism):
    """
    Insecure mplementation of the Permute-And-Flip Mechanism. This mechanism can be used
    once after which its privacy budget will be exhausted and it can no longer be used.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        utility_function: the utility function. This should accpet an array of values
                          produced by the query function and return an 1D array of
                          utilities of the same size as output_range.
        sensitivity: the sensitivity of the utility function.
        output_range: an array of possible output values for the mechanism.
    """

    def __init__(
        self,
        epsilon,
        utility_function,
        sensitivity,
        output_range,
    ):
        super(PermuteAndFlipMechanism, self).__init__(epsilon)
        self.utility_function = utility_function
        self.sensitivity = sensitivity
        self.output_range = output_range

        self.effective_epsilon = self.epsilon / (2.0 * sensitivity)

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy array of the output of a query.

        Returns:
            An element of output_range set of the mechanism.
        """

        self._check_valid()
        self._is_valid = False
        self._update_accountant()

        utilities = self.utility_function(values)
        index = backend.permute_and_flip_mechanism(utilities, self.effective_epsilon)
        return self.output_range[index]

    @property
    def privacy_consumed(self):
        """
        Computes the privacy budget consumed by the mechanism so far.
        """
        if self._is_valid:
            return 0
        else:
            return self.epsilon
