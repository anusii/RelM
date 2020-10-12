import crlibm
import math
import struct

import numpy as np

from . import samplers
from differential_privacy import backend


class ReleaseMechanism:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.cutoff = 1
        self.current_count = 0

    def _is_valid(self):
        return self.current_count < self.cutoff

    def release(self):
        raise NotImplementedError()


# spec = [
#     ('epsilon', float64),
#     ('cutoff', int64),
#     ('current_count', int64)
# ]
# @jitclass(spec)
class LaplaceMechanism(ReleaseMechanism):
    def release(self, values, sensitivity=1):
        if self._is_valid():
            self.current_count += 1
            n = len(values)
            b = sensitivity / self.epsilon
            perturbations = samplers.laplace(n, b)
            perturbed_values = values + perturbations
        else:
            raise RuntimeError()

        return perturbed_values


# @jitclass(spec)
class GeometricMechanism(ReleaseMechanism):
    def release(self, values):
        if self._is_valid():
            self.current_count += 1
            n = len(values)
            q = 1.0 / np.exp(self.epsilon)
            perturbations = samplers.two_sided_geometric(n, q)
            perturbed_values = values + perturbations
        else:
            raise RuntimeError()

        return perturbed_values


# class Sparse(ReleaseMechanism):
#     def __init__(self, epsilon, threshold, cutoff):
#         super(Sparse, self).__init__(epsilon)
#         self.threshold = threshold
#         self.cutoff = cutoff
#
#     def next_above_threshold(self, values):
#         n = len(values)
#         threshold_perturbation = samplers.laplace(1, b=2.0 * self.cutoff / self.epsilon)
#         perturbed_threshold = self.threshold + threshold_perturbation
#         value_perturbations = samplers.laplace(n, b=4.0 * self.cutoff / self.epsilon)
#         perturbed_values = values + value_perturbations
#         indicators = perturbed_values > perturbed_threshold
#         if np.any(indicators):
#             index = np.argmax(indicators)
#         else:
#             index = None
#         return index
#
#     def all_above_threshold(self, values):
#         threshold = self.threshold + samplers.laplace(
#             1, b=2.0 * self.cutoff / self.epsilon
#         )
#         return backend.all_above_threshold(
#             values, 4.0 * self.cutoff / self.epsilon, threshold
#         )
#
#     def release(self, values):
#         if self._is_valid():
#             remaining = self.cutoff - self.current_count
#             indices = self.all_above_threshold(values)
#             indices = indices[:remaining]
#             self.current_count += len(indices)
#         else:
#             raise RuntimeError()
#
#         return indices
#
#
# class AboveThreshold(Sparse):
#     def __init__(self, epsilon, threshold):
#         super(AboveThreshold, self).__init__(epsilon, threshold, 1)
#
#
# class SparseNumeric(Sparse):
#     def __init__(self, epsilon, threshold, cutoff):
#         super(SparseNumeric, self).__init__(epsilon, threshold, cutoff)
#         self.epsilon = (8 / 9) * epsilon
#         self.epsilon2 = (2 / 9) * epsilon
#
#     def release(self, values):
#         if self._is_valid():
#             remaining = self.cutoff - self.current_count
#             indices = self.all_above_threshold(values)
#             indices = indices[:remaining]
#             self.current_count += len(indices)
#             sliced_values = values[indices]
#             n = len(sliced_values)
#             b = 2 * self.cutoff / self.epsilon2
#             perturbations = samplers.laplace(n, b)
#             perturbed_values = sliced_values + perturbations
#         else:
#             raise RuntimeError()
#
#         return indices, perturbed_values


class SparseGeneric(ReleaseMechanism):
    def __init__(self, epsilon1, epsilon2, epsilon3, threshold, cutoff):
        epsilon = epsilon1 + epsilon2 + epsilon3
        self.epsilon = epsilon
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.threshold = threshold
        self.cutoff = cutoff
        self.current_count = 0

    def all_above_threshold(self, values):
        rho = samplers.laplace(1, b=1.0 / self.epsilon1)
        threshold = self.threshold + rho
        return backend.all_above_threshold(
            values, 2.0 * self.cutoff / self.epsilon2, threshold
        )

    def release(self, values):
        if self._is_valid():
            remaining = self.cutoff - self.current_count
            indices = self.all_above_threshold(values)
            indices = indices[:remaining]
            self.current_count += len(indices)
            sliced_values = values[indices]
            n = len(sliced_values)
            if self.epsilon3 > 0:
                b = self.cutoff / self.epsilon3
                perturbations = samplers.laplace(n, b)
                perturbed_values = sliced_values + perturbations
            else:
                perturbed_values = np.array([np.nan for i in range(n)])
        else:
            raise RuntimeError()

        return indices, perturbed_values


class SparseNumeric(SparseGeneric):
    def __init__(self, epsilon, threshold, cutoff, e2_weight=None, e3_weight=None):
        e1_weight = 1.0
        if e2_weight is None:
            e2_weight = (2.0 * cutoff) ** (2.0 / 3.0)
        if e3_weight is None:
            e3_weight = e1_weight + e2_weight
        epsilon_weights = (e1_weight, e2_weight, e3_weight)
        total_weight = sum(epsilon_weights)
        epsilon1 = (epsilon_weights[0] / total_weight) * epsilon
        epsilon2 = (epsilon_weights[1] / total_weight) * epsilon
        epsilon3 = (epsilon_weights[2] / total_weight) * epsilon
        super(SparseNumeric, self).__init__(
            epsilon1, epsilon2, epsilon3, threshold, cutoff
        )


class SparseIndicator(SparseNumeric):
    def __init__(self, epsilon, threshold, cutoff, e2_weight=None):
        e3_weight = 0.0
        super(SparseIndicator, self).__init__(
            epsilon, threshold, cutoff, e2_weight, e3_weight
        )

    def release(self, values):
        indices, perturbed_values = super(SparseIndicator, self).release(values)
        return indices


class AboveThreshold(SparseIndicator):
    def __init__(self, epsilon, threshold, e2_weight=None):
        cutoff = 1
        super(AboveThreshold, self).__init__(epsilon, threshold, cutoff, e2_weight)

    def release(self, values):
        indices = super(AboveThreshold, self).release(values)
        return int(indices[0])


class Snapping(ReleaseMechanism):
    def __init__(self, epsilon, B):
        lam = (1 + 2 ** (-49) * B) / epsilon
        if (B <= lam) or (B >= (2 ** 46 * lam)):
            raise ValueError()
        self.lam = lam
        self.quanta = 2 ** math.ceil(math.log2(self.lam))
        self.B = B
        super(Snapping, self).__init__(epsilon)

    def release(self, values):
        if self._is_valid():
            self.current_count += 1
            release_values = backend.snapping(values, self.B, self.lam, self.quanta)
        else:
            raise RuntimeError

        return release_values
