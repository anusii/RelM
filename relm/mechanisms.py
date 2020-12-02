import math
import numpy as np
import secrets
from relm import backend


class ReleaseMechanism:
    def __init__(self, epsilon):
        self.epsilon = epsilon
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
        raise NotImplementedError()


class LaplaceMechanism(ReleaseMechanism):
    """
    Secure implementation of the Laplace mechanism. This mechanism can be used once
    after which its privacy budget will be exhausted and it can no longer be used.

    Under the hood this mechanism samples exactly from a 64-bit fixed point
    Laplace mechanism bit by bit.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        sensitivity: the sensitivity of the query to which this mechanism will be applied.
        precision: number of fractional bits to use in the internal fixed point representation.
    """

    def __init__(self, epsilon, sensitivity, precision):
        self.sensitivity = sensitivity
        self.precision = precision
        super(LaplaceMechanism, self).__init__(epsilon)

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy array of the output of a query.

        Returns:
            A numpy array of perturbed values.
        """

        self._check_valid()
        self._is_valid = False
        self._update_accountant()
        args = (values, self.sensitivity, self.epsilon, self.precision)
        return backend.laplace_mechanism(*args)

    @property
    def privacy_consumed(self):
        """
        Computes the privacy budget consumed by the mechanism so far.
        """
        if self._is_valid:
            return 0
        else:
            return self.epsilon


class GeometricMechanism(LaplaceMechanism):
    """
    Secure implementation of the Geometric mechanism. This mechanism can be used once
    after which its privacy budget will be exhausted and it can no longer be used.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        sensitivity: the sensitivity of the query to which this mechanism will be applied.
    """

    def __init__(self, epsilon, sensitivity):
        super(GeometricMechanism, self).__init__(epsilon, sensitivity, precision=0)

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy array of the output of a query.

        Returns:
            A numpy array of perturbed values.
        """
        self._check_valid()
        self._is_valid = False
        self._update_accountant()
        return backend.geometric_mechanism(values, self.sensitivity, self.epsilon)


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
                utilities, self.sensitivity, self.epsilon
            )
        elif value == "gumbel_trick":
            sampler = lambda utilities: backend.exponential_mechanism_gumbel_trick(
                utilities, self.sensitivity, self.epsilon
            )
        elif value == "sample_and_flip":
            sampler = lambda utilities: backend.exponential_mechanism_sample_and_flip(
                utilities, self.sensitivity, self.epsilon
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
        index = backend.permute_and_flip_mechanism(
            utilities, self.sensitivity, self.epsilon
        )
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
        self.epsilon = epsilon
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.cutoff = cutoff
        self.monotonic = monotonic
        self.precision = precision
        self.current_count = 0

        temp = np.array([threshold], dtype=np.float64)
        args = (temp, sensitivity, epsilon1, precision)
        self.perturbed_threshold = backend.laplace_mechanism(*args)[0]
        super(SparseGeneric, self).__init__(epsilon)

    def all_above_threshold(self, values):
        if self.monotonic:
            b = (self.sensitivity * self.cutoff) / self.epsilon2
        else:
            b = (2.0 * self.sensitivity * self.cutoff) / self.epsilon2
        return backend.all_above_threshold(
            values, b, self.perturbed_threshold, self.precision
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
            temp = self.sensitivity * self.cutoff
            args = (sliced_values, temp, self.epsilon3, self.precision)
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


class SnappingMechanism(ReleaseMechanism):
    """
    Secure implementation of the Snapping mechanism. This mechanism can be used once
    after which its privacy budget will be exhausted and it can no longer be used.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        B: the bound of the range to use for the snapping mechanism.
            B should ideally be larger than the range of outputs expected but the larger B is
            the less accurate the results.
    """

    def __init__(self, epsilon, B):
        lam = (1 + 2 ** (-49) * B) / epsilon
        if (B <= lam) or (B >= (2 ** 46 * lam)):
            raise ValueError()
        self.lam = lam
        self.quanta = 2 ** math.ceil(math.log2(self.lam))
        self.B = B
        super(SnappingMechanism, self).__init__(epsilon)

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy array of the output of a query.

        Returns:
            A numpy array of perturbed values.
        """
        self._check_valid()
        args = (values, self.B, self.lam, self.quanta)
        release_values = backend.snapping(*args)
        self._is_valid = False
        self._update_accountant()
        return release_values

    @property
    def privacy_consumed(self):
        """
        Computes the privacy budget consumed by the mechanism so far.
        """
        if self._is_valid:
            return 0
        else:
            return self.epsilon


class ReportNoisyMax(ReleaseMechanism):
    """
    Secure implementation of the ReportNoisyMax mechanism. This mechanism can be used
    once after which its privacy budget will be exhausted and it can no longer be used.

    This mechanism adds Laplace noise to each of a set of counting queries and
    returns both the index of the largest perturbed value (the argmax) and the
    largest perturbed value.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        precision: number of fractional bits to use in the internal fixed point representation.
    """

    def __init__(self, epsilon, precision):
        self.sensitivity = 1.0
        self.precision = precision
        super(ReportNoisyMax, self).__init__(epsilon)

    def release(self, values):
        """
        Releases a differential private query response.

        Args:
            values: numpy array of the outputs of a collection of counting queries.

        Returns:
            A tuple containing the (argmax, max) of the perturbed values.
        """
        self._check_valid()
        self._is_valid = False
        self._update_accountant()
        args = (values, self.sensitivity, self.epsilon, self.precision)
        perturbed_values = backend.laplace_mechanism(*args)
        valmax = np.max(perturbed_values)
        argmax = secrets.choice(np.where(perturbed_values == valmax)[0])
        return (argmax, valmax)

    @property
    def privacy_consumed(self):
        """
        Computes the privacy budget consumed by the mechanism so far.
        """
        if self._is_valid:
            return 0
        else:
            return self.epsilon


class MultiplicativeWeights(ReleaseMechanism):
    """
    Secure implementation of the private Multiplicative Weights mechanism.
    This mechanism can be used to answer multiple linear queries.

    Args:
        epsilon: the privacy parameter to use
        cutoff: the number of queries that access the private data
        threshold: the error tolerable
        learning_rate: the learning rate of the multiplicative weights algorithm
        data: a 1D numpy array of the underlying database
    """

    def __init__(self, epsilon, cutoff, threshold, learning_rate, data):
        super(MultiplicativeWeights, self).__init__(epsilon)
        self.data = data
        self.learning_rate = learning_rate
        self.sparse_numeric = SparseNumeric(
            epsilon, sensitivity=1, threshold=threshold, cutoff=cutoff
        )
        self.data_est = np.ones(len(data)) / len(data)

        # this assumes that the l1 norm of the database is public
        self.l1_norm = data.sum()

    @property
    def privacy_consumed(self):
        return self.sparse_numeric.privacy_consumed

    def update_weights(self, est_answer, noisy_answer, query):
        if noisy_answer < est_answer:
            r = query
        else:
            r = 1 - query

        self.data_est *= np.exp(-r * self.learning_rate)
        self.data_est /= self.data_est.sum()

    def release(self, queries):
        """
        Returns private answers to the queries.

        Args:
            queries: a list of queries as 1D 1/0 indicator numpy arrays
        Returns:
            a numpy array of the private query responses
        """

        results = []
        for query in queries:
            true_answer = (query * self.data).sum()
            est_answer = (query * self.data_est).sum() * self.l1_norm

            error = true_answer - est_answer
            errors = np.array([error, -error])
            indices, release_values = self.sparse_numeric.release(errors)

            if len(indices) == 0:
                results.append(est_answer)
            else:
                noisy_answer = est_answer + (1 - 2 * indices[0]) * release_values[0]
                results.append(noisy_answer)

        return np.array(results)
