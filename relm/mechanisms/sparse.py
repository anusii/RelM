from .base import ReleaseMechanism
import numpy as np
from relm import backend


class SparseGeneric(ReleaseMechanism):
    def __init__(
        self,
        epsilon1,
        epsilon2,
        epsilon3,
        sensitivity,
        threshold,
        cutoff,
        monotonic,
        precision=35,
    ):
        epsilon = epsilon1 + epsilon2 + epsilon3
        super(SparseGeneric, self).__init__(epsilon)

        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.cutoff = cutoff
        self.monotonic = monotonic
        self.precision = precision
        self.current_count = 0

        self.effective_epsilon1 = self.epsilon1 / self.sensitivity
        self.effective_epsilon2 = self.epsilon2 / (self.cutoff * self.sensitivity)
        if not self.monotonic:
            self.effective_epsilon2 /= 2.0
        self.effective_epsilon3 = self.epsilon3 / (self.cutoff * self.sensitivity)

        temp = np.array([self.threshold], dtype=np.float64)
        args = (temp, self.effective_epsilon1, self.precision)
        self.perturbed_threshold = backend.laplace_mechanism(*args)[0]

    def all_above_threshold(self, values):
        return backend.all_above_threshold(
            values, self.effective_epsilon2, self.perturbed_threshold, self.precision
        )

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy arrays of query responses. Each element is the response to a different query.

        Returns:
            A tuple of numpy arrays containing the perturbed values and the corresponding indices.
        """
        self._check_valid()

        remaining = self.cutoff - self.current_count
        indices = self.all_above_threshold(values)
        indices = indices[:remaining]
        self.current_count += len(indices)

        if self.current_count == self.cutoff:
            self._is_valid = False

        self._update_accountant()

        if self.epsilon3 > 0:
            sliced_values = values[indices]
            args = (
                sliced_values,
                self.effective_epsilon3,
                self.precision,
            )
            release_values = backend.laplace_mechanism(*args)
            return indices, release_values
        else:
            return indices

    @property
    def privacy_consumed(self):
        """
        Computes the privacy budget consumed by the mechanism so far.
        """
        if self._is_valid:
            return self.epsilon1 + (self.current_count / self.cutoff) * (
                self.epsilon2 + self.epsilon3
            )
        else:
            return self.epsilon


class SparseNumeric(SparseGeneric):
    """
    Secure implementation of the SparseNumeric mechanism.
    This mechanism can used repeatedly until `cutoff` positive queries have been answered
    after which the mechanism is exhausted and cannot be used.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        sensitivity: the sensitivity of the query function
        threshold: the threshold to use
        cutoff: the number of positive queries that can be answered
        e2_weight: the relative amount of the privacy budget to allocate to
            perturbing the answers for comparison. If set to None (default) this will be
            auto-calculated.
        e3_weight: the relative amount of the privacy budget to allocate to
            perturbing the answers for release. If set to None (default) this will be
            auto-calculated.
        monotonic: boolean indicating whether the queries are monotonic.
    """

    def __init__(
        self,
        epsilon,
        sensitivity,
        threshold,
        cutoff,
        e2_weight=None,
        e3_weight=None,
        monotonic=False,
        precision=35,
    ):
        e1_weight = 1.0
        if e2_weight is None:
            if monotonic:
                e2_weight = (cutoff) ** (2.0 / 3.0)
            else:
                e2_weight = (2.0 * cutoff) ** (2.0 / 3.0)
        if e3_weight is None:
            e3_weight = e1_weight + e2_weight
        epsilon_weights = (e1_weight, e2_weight, e3_weight)
        total_weight = sum(epsilon_weights)
        epsilon1 = (epsilon_weights[0] / total_weight) * epsilon
        epsilon2 = (epsilon_weights[1] / total_weight) * epsilon
        epsilon3 = (epsilon_weights[2] / total_weight) * epsilon
        super(SparseNumeric, self).__init__(
            epsilon1,
            epsilon2,
            epsilon3,
            sensitivity,
            threshold,
            cutoff,
            monotonic,
            precision,
        )


class SparseIndicator(SparseNumeric):
    """
    Secure implementation of the SparseIndicator mechanism.
    This mechanism can used repeatedly until `cutoff` positive queries have been answered
    after which the mechanism is exhausted and cannot be used.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        sensitivity: the sensitivity of the query function
        threshold: the threshold to use
        cutoff: the number of positive queries that can be answered
        e2_weight: the relative amount of the privacy budget to allocate to
            perturbing the answers for comparison. If set to None (default) this will be
            auto-calculated.
        monotonic: boolean indicating whether the queries are monotonic.
    """

    def __init__(
        self,
        epsilon,
        sensitivity,
        threshold,
        cutoff,
        e2_weight=None,
        monotonic=False,
        precision=35,
    ):
        e3_weight = 0.0
        super(SparseIndicator, self).__init__(
            epsilon,
            sensitivity,
            threshold,
            cutoff,
            e2_weight,
            e3_weight,
            monotonic,
            precision,
        )

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy arrays of query responses. Each element is the response to a different query.

        Returns:
            A tuple of numpy arrays containing the indices of noisy queries above the threshold.
        """
        return super(SparseIndicator, self).release(values)


class AboveThreshold(SparseIndicator):
    """
    Secure implementation of the AboveThreshold mechanism. This returns the index
    of the first query above the threshold after which the mechanism will be exhausted.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        sensitivity: the sensitivity of the query function
        threshold: the threshold to use
        e2_weight: the relative amount of the privacy budget to allocate to
            perturbing the answers for comparison. If set to None (default) this will be
            auto-calculated.
        monotonic: boolean indicating whether the queries are monotonic.
    """

    def __init__(
        self,
        epsilon,
        sensitivity,
        threshold,
        e2_weight=None,
        monotonic=False,
        precision=35,
    ):
        cutoff = 1
        super(AboveThreshold, self).__init__(
            epsilon, sensitivity, threshold, cutoff, e2_weight, monotonic, precision
        )

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy arrays of query responses. Each element is the response to a different query.

        Returns:
            The index of the first noisy query above the threshold.
        """
        indices = super(AboveThreshold, self).release(values)
        if len(indices) > 0:
            index = int(indices[0])
        else:
            index = None
        return index
