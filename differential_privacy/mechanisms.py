import crlibm
import math
import struct

import numpy as np

from . import samplers
from differential_privacy import backend


class ReleaseMechanism:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.privacy_loss = 0

    def release(self, data):
        if self.privacy_loss < self.epsilon:
            return self.unsafe_release(data)

        raise RuntimeError

    def unsafe_release(self):
        raise NotImplementedError()


class LaplaceMechanism(ReleaseMechanism):
    def __init__(self, epsilon, sensitivity, precision):
        self.sensitivity = sensitivity
        self.precision = precision
        super(LaplaceMechanism, self).__init__(epsilon)

    def unsafe_release(self, values):

        self.privacy_loss += self.epsilon
        n = len(values)
        b = (self.sensitivity + 2 ** (-self.precision)) / self.epsilon
        fp_perturbations = samplers.fixed_point_laplace(n, b, self.precision)
        fp_values = np.rint(values * 2 ** self.precision).astype(np.int64)
        temp = (fp_values + fp_perturbations).astype(np.float64)
        perturbed_values = temp * 2 ** (-self.precision)

        return perturbed_values


class GeometricMechanism(ReleaseMechanism):
    def unsafe_release(self, values):
        self.privacy_loss += self.epsilon
        n = len(values)
        q = 1.0 / np.exp(self.epsilon)
        perturbations = samplers.two_sided_geometric(n, q)
        perturbed_values = values + perturbations

        return perturbed_values


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
    ):
        epsilon = epsilon1 + epsilon2 + epsilon3
        self.epsilon = epsilon
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.rho = samplers.laplace(1, b=sensitivity / epsilon1)
        self.cutoff = cutoff
        self.monotonic = monotonic
        self.current_count = 0
        super(SparseGeneric, self).__init__(epsilon)

    def all_above_threshold(self, values):
        threshold = self.threshold + self.rho
        if self.monotonic:
            b = (self.sensitivity * self.cutoff) / self.epsilon2
        else:
            b = (2.0 * self.sensitivity * self.cutoff) / self.epsilon2
        return backend.all_above_threshold(values, b, threshold)

    def unsafe_release(self, values):
        if self.privacy_loss == 0:
            self.privacy_loss += self.epsilon1

        remaining = self.cutoff - self.current_count
        indices = self.all_above_threshold(values)
        indices = indices[:remaining]
        n = len(indices)
        self.current_count += n
        self.privacy_loss += self.epsilon2 * n / self.cutoff

        if self.epsilon3 > 0:
            sliced_values = values[indices]
            b = (self.sensitivity * self.cutoff) / self.epsilon3

            perturbations = samplers.laplace(n, b)
            perturbed_values = sliced_values + perturbations
            self.privacy_loss += self.epsilon3 * n / self.cutoff

            return indices, perturbed_values
        else:
            return indices


class SparseNumeric(SparseGeneric):
    def __init__(
        self,
        epsilon,
        sensitivity,
        threshold,
        cutoff,
        e2_weight=None,
        e3_weight=None,
        monotonic=False,
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
            epsilon1, epsilon2, epsilon3, sensitivity, threshold, cutoff, monotonic
        )


class SparseIndicator(SparseNumeric):
    def __init__(
        self, epsilon, sensitivity, threshold, cutoff, e2_weight=None, monotonic=False
    ):
        e3_weight = None
        super(SparseIndicator, self).__init__(
            epsilon, sensitivity, threshold, cutoff, e2_weight, e3_weight, monotonic
        )

    def unsafe_release(self, values):
        (indices, *_) = super(SparseIndicator, self).unsafe_release(values)
        return indices


class AboveThreshold(SparseIndicator):
    def __init__(
        self, epsilon, sensitivity, threshold, e2_weight=None, monotonic=False
    ):
        cutoff = 1
        super(AboveThreshold, self).__init__(
            epsilon, sensitivity, threshold, cutoff, e2_weight, monotonic
        )

    def unsafe_release(self, values):
        indices = super(AboveThreshold, self).unsafe_release(values)
        if len(indices) > 0:
            index = int(indices[0])
        else:
            index = None
        return index


class Snapping(ReleaseMechanism):
    def __init__(self, epsilon, B):
        lam = (1 + 2 ** (-49) * B) / epsilon
        if (B <= lam) or (B >= (2 ** 46 * lam)):
            raise ValueError()
        self.lam = lam
        self.quanta = 2 ** math.ceil(math.log2(self.lam))
        self.B = B
        super(Snapping, self).__init__(epsilon)

    def unsafe_release(self, values):
        self.privacy_loss += self.epsilon
        release_values = backend.snapping(values, self.B, self.lam, self.quanta)
        return release_values
