import numpy as np
from numba import jitclass, float64, int64

from . import samplers

class ReleaseMechanism:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.cutoff = 1
        self.current_count = 0

    def _is_valid(self):
        return (self.current_count < self.cutoff)

    def release(self):
        raise NotImplementedError()


# spec = [
#     ('epsilon', float64),
#     ('cutoff', int64),
#     ('current_count', int64)
# ]
# @jitclass(spec)
class LaplaceMechanism(ReleaseMechanism):
    def release(self, values, sensitivity):
        if self._is_valid():
            self.current_count += 1
            n = len(values)
            b = sensitivity / self.epsilon
            perturbations = samplers.laplace(n, b)
            perturbed_values = values + perturbations
        else:
            raise RuntimeError()

        return perturbed_values

#@jitclass(spec)
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
        threshold_perturbation = samplers.laplace(1, b=2.0*self.cutoff/self.epsilon)
        perturbed_threshold = self.threshold + threshold_perturbation
        value_perturbations = samplers.laplace(n, b=4.0*self.cutoff/self.epsilon)
        perturbed_values = values + value_perturbations
        indicators = perturbed_values > perturbed_threshold
        if np.any(indicators):
            index = np.argmax(indicators)
        else:
            index = None
        return index

    def all_above_threshold(self, values):
        indices = []
        current_start = 0
        while current_start < len(values):
            index = self.next_above_threshold(values[current_start:])
            if index is not None:
                indices.append(current_start + index)
                current_start += index + 1
            else:
                break
        return np.array(indices)

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
        self.epsilon = (8/9)*epsilon
        self.epsilon2 = (2/9)*epsilon

    def release(self, values):
        if self._is_valid():
            remaining = self.cutoff - self.current_count
            indices = self.all_above_threshold(values)
            indices = indices[:remaining]
            self.current_count += len(indices)
            sliced_values = values[indices]
            n = len(sliced_values)
            b = 2*self.cutoff / self.epsilon2
            perturbations = samplers.laplace(n,b)
            perturbed_values = sliced_values + perturbations
        else:
            raise RuntimeError()

        return indices, perturbed_values
