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
        self.Lam = 2 ** math.ceil(math.log2(self.lam))
        self.B = B
        super(Snapping, self).__init__(epsilon)

    def clamp(self, x):
        n = x.size
        clamp_vals = self.B * np.ones(n)
        ret = np.sign(x) * np.min((np.abs(x), clamp_vals), axis=0)
        return ret

    def round_pow2(self, x):
        ret = self.Lam * np.round((x / self.Lam))
        return ret

    def double_to_uint64(self, x):
        s = struct.pack(">d", x)
        i = struct.unpack(">Q", s)[0]
        return i

    def release(self, values):
        if self._is_valid():
            self.current_count += 1
            n = len(values)
            unifs = samplers.uniform_double(n)
            log_unifs = np.array([crlibm.log_rn(u) for u in unifs])
            perturbations = self.lam * log_unifs
            sgn = np.sign(samplers.uniform(n, a=-0.5, b=0.5))
            perturbed_values = self.clamp(values) + sgn * perturbations
            rounded_values = self.round_pow2(perturbed_values)
            release_values = self.clamp(rounded_values)
        else:
            raise RuntimeError

        return release_values
