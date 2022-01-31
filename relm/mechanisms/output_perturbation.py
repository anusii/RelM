from .base import ReleaseMechanism
import numpy as np
from relm import backend
import secrets
import math

import scipy.stats


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

    def __init__(self, epsilon, sensitivity, precision=35):
        super(LaplaceMechanism, self).__init__(epsilon)
        self.sensitivity = sensitivity
        self.precision = precision
        self.effective_epsilon = self.epsilon / (
            self.sensitivity + 2.0 ** -self.precision
        )

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
        args = (values, self.effective_epsilon, self.precision)
        return backend.laplace_mechanism(*args)


class GeometricMechanism(ReleaseMechanism):
    """
    Secure implementation of the Geometric mechanism. This mechanism can be used once
    after which its privacy budget will be exhausted and it can no longer be used.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        sensitivity: the sensitivity of the query to which this mechanism will be applied.
    """

    def __init__(self, epsilon, sensitivity):
        super(GeometricMechanism, self).__init__(epsilon)
        self.sensitivity = sensitivity
        self.effective_epsilon = self.epsilon / self.sensitivity

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
        return backend.geometric_mechanism(values, self.effective_epsilon)


class DiscreteGaussianMechanism(ReleaseMechanism):
    """
    Secure implementation of the Discrete Gaussian mechanism. This mechanism can
    be used once after which its privacy budget will be exhausted and it can no
    longer be used.

    Under the hood this mechanism samples a from a discrete Laplace distribution
    supported on the integers and then uses rejectino sampling to produce samples
    from the desired Discrete Gaussian distribution.

    Args:
        epsilon: the maximum multiplicative privacy loss of the mechanism.
        delta: the maximum additive privacy loss of the mechanism.
        sensitivity: the sensitivity of the query to which this mechanism will be applied.
    """

    def __init__(self, epsilon, delta, sensitivity):
        super(DiscreteGaussianMechanism, self).__init__(epsilon)
        self.delta = delta
        self.sensitivity = sensitivity
        self.effective_epsilon = self.epsilon / self.sensitivity

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

        return backend.discrete_gaussian_mechanism(
            values, self.effective_epsilon, self.delta
        )


class GaussianMechanism(ReleaseMechanism):
    """
    Secure implementation of the Gaussian mechanism. This mechanism can be used once
    after which its privacy budget will be exhausted and it can no longer be used.

    Under the hood this mechanism samples a from a discrete Gaussian distribution
    supported on the integers and then rescales the result to get a fixed-point
    approximation to the real-valued Gaussian distribution.

    Args:
        epsilon: the maximum multiplicative privacy loss of the mechanism.
        delta: the maximum additive privacy loss of the mechanism.
        sensitivity: the sensitivity of the query to which this mechanism will be applied.
        precision: number of fractional bits to use in the internal fixed point representation.
    """

    def __init__(self, epsilon, delta, sensitivity, precision=35):
        super(GaussianMechanism, self).__init__(epsilon)
        self.delta = delta
        self.sensitivity = sensitivity
        self.precision = precision
        self.effective_epsilon = self.epsilon / (
            self.sensitivity + 2.0 ** -self.precision
        )

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

        return backend.gaussian_mechanism(
            values, self.effective_epsilon, self.delta, self.precision
        )


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


class CauchyMechanism(ReleaseMechanism):
    """
    ***Insecure*** implementation of the Cauchy mechanism. This mechanism can be used
    once after which its privacy budget will be exhausted and it can no longer be used.

    Args:
        epsilon: the maximum privacy loss of the mechanism.
        beta: the smoothness parameter for the beta-smooth upper bound on the local
              sensitivity of the query to which this mechanism will be applied.
    """

    def __init__(self, epsilon, beta):
        super(CauchyMechanism, self).__init__(epsilon)
        self.beta = beta
        if self.beta > self.epsilon / 6.0:
            raise ValueError("beta must not be greater than epsilon/6.0.")

    def release(self, values, smooth_sensitivity):
        """
        Releases a differential private query response.

        Args:
            values: numpy array of the output of a query.
            smooth_senstivity: A beta-smooth upper bound on the local sensitivity of
                               a query.

        Returns:
            A numpy array of perturbed values.
        """

        self._check_valid()
        self._is_valid = False
        self._update_accountant()
        effective_epsilon = self.epsilon / (6.0 * smooth_sensitivity)
        return backend.cauchy_mechanism(values, effective_epsilon)


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

    def __init__(self, epsilon, precision=35):
        super(ReportNoisyMax, self).__init__(epsilon)
        self.sensitivity = 1.0
        self.precision = precision
        self.effective_epsilon = self.epsilon / self.sensitivity

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
        args = (values, self.effective_epsilon, self.precision)
        perturbed_values = backend.laplace_mechanism(*args)
        valmax = np.max(perturbed_values)
        argmax = secrets.choice(np.where(perturbed_values == valmax)[0])
        return (argmax, valmax)
