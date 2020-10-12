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

    PRECISION = 2.0 ** (-32)

    def release(self, values, sensitivity=1):
        if self._is_valid():
            self.current_count += 1
            n = len(values)
            sensitivity = (sensitivity + self.PRECISION) / self.PRECISION
            q = 1.0 / np.exp(self.epsilon / sensitivity)
            perturbations = samplers.geometric(n, q).astype(float) * sensitivity
            perturbed_values = (
                backend.round_array(values, self.PRECISION) + perturbations
            )
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


class Sparse(ReleaseMechanism):
    def __init__(self, epsilon, threshold, cutoff):
        super(Sparse, self).__init__(epsilon)
        self.threshold = threshold
        self.cutoff = cutoff

    def next_above_threshold(self, values):
        n = len(values)
        threshold_perturbation = samplers.laplace(1, b=2.0 * self.cutoff / self.epsilon)
        perturbed_threshold = self.threshold + threshold_perturbation
        value_perturbations = samplers.laplace(n, b=4.0 * self.cutoff / self.epsilon)
        perturbed_values = values + value_perturbations
        indicators = perturbed_values > perturbed_threshold
        if np.any(indicators):
            index = np.argmax(indicators)
        else:
            index = None
        return index

    def all_above_threshold(self, values):
        threshold = self.threshold + samplers.laplace(
            1, b=2.0 * self.cutoff / self.epsilon
        )
        return backend.all_above_threshold(
            values, 4.0 * self.cutoff / self.epsilon, threshold
        )

    def release(self, values):
        if self._is_valid():
            remaining = self.cutoff - self.current_count
            indices = self.all_above_threshold(values)
            indices = indices[:remaining]
            self.current_count += len(indices)
        else:
            raise RuntimeError()

        return indices


class AboveThreshold(Sparse):
    def __init__(self, epsilon, threshold):
        super(AboveThreshold, self).__init__(epsilon, threshold, 1)


class SparseNumeric(Sparse):
    def __init__(self, epsilon, threshold, cutoff):
        super(SparseNumeric, self).__init__(epsilon, threshold, cutoff)
        self.epsilon = (8 / 9) * epsilon
        self.epsilon2 = (2 / 9) * epsilon

    def release(self, values):
        if self._is_valid():
            remaining = self.cutoff - self.current_count
            indices = self.all_above_threshold(values)
            indices = indices[:remaining]
            self.current_count += len(indices)
            sliced_values = values[indices]
            n = len(sliced_values)
            b = 2 * self.cutoff / self.epsilon2
            perturbations = samplers.laplace(n, b)
            perturbed_values = sliced_values + perturbations
        else:
            raise RuntimeError()

        return indices, perturbed_values


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
