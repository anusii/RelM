import math
import numpy as np
from differential_privacy import backend
from numba import objmode, uint8, njit

BITS_PER_FLOAT = 53
BYTES_PER_FLOAT = math.ceil(BITS_PER_FLOAT / 8)
SHIFT_AMOUNT = 8 * BYTES_PER_FLOAT - BITS_PER_FLOAT
NORMALIZER = 2 ** -BITS_PER_FLOAT


@njit
def _int_from_bytes(bs):
    """
    Converts bytes into an integer.

    :param bs: a bytes object to converted into an integer.
    :return: an big-endian integer representation of the input bytes.
    """
    accumulator = 0
    for i, b in enumerate(bs[::-1]):
        accumulator += b * (256) ** i
    return accumulator


@njit
def _significands_from_bytes(bs, shift_amount=SHIFT_AMOUNT):
    n = len(bs) // BYTES_PER_FLOAT
    significands = np.zeros(n, dtype=np.int64)
    for i in range(n):
        start_index = BYTES_PER_FLOAT * i
        stop_index = BYTES_PER_FLOAT * (i + 1)
        significands[i] = _int_from_bytes(bs[start_index:stop_index]) >> shift_amount
    return significands


@njit
def _floats_from_bytes(bs):
    """
    Converts bytes into floats in the interval [0,1).

    :param bs: a bytes object to be converted into floats.
    :return: a 1D numpy array of length `bs//BYTES_PER_FLOAT` of floats in the interval [0,1).
    """
    n = len(bs) // BYTES_PER_FLOAT
    significands = _significands_from_bytes(bs)
    unifs = significands * NORMALIZER
    return unifs


def uniform(n, a=0.0, b=1.0):
    """
    Samples from the uniform distribution on the interval [a,b).

    :param n: the number of samples to draw.
    :param a: the lower bound of the interval to draw from.
    :param b: the upper bound of the interval to draw from.
    :return: a 1D numpy array of length `n` of samples from the Uniform(0,1) distribution.
    """
    unifs_01 = backend.uniform(n)
    unifs_ab = (b - a) * unifs_01 + a
    return unifs_ab


def exponential(n, b):
    """
    Samples from the Exponential(b) distribution.

    :param n: the number of samples to draw.
    :param b: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the Exponential(b) distribution.
    """
    return backend.exponential(b, n)


def laplace(n, b=1):
    """
    Samples from the Laplace(b) distribution.

    :param n: the number of samples to draw.
    :param b: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the Laplace(b) distribution.
    """
    return backend.laplace(b, n)


def geometric(n, p=0.5):
    """
    Samples from the Geometric(p) distribution.

    :param n: the number of samples to draw.
    :param p: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the Geometric(p) distribution.
    """
    return backend.geometric(p, n).astype(np.int64)


def two_sided_geometric(n, q=0.5):
    """
    Samples from the TwoSidedGeometric(q) distribution.

    :param n: the number of samples to draw.
    :param q: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the TwoSidedGeometric(q) distribution.
    """
    return backend.two_sided_geometric(q, n).astype(np.int64)


def simple_laplace(n, b=1):
    """
    Samples from the Laplace(b) distribution.

    :param n: the number of samples to draw.
    :param b: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the Laplace(b) distribution.
    """
    xs = [exponential(n, b) for i in range(2)]
    ys = xs[0] - xs[1]
    return ys


def simple_two_sided_geometric(n, q=0.5):
    """
    Samples from the TwoSidedGeometric(q) distribution.

    :param n: the number of samples to draw.
    :param q: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the TwoSidedGeometric(q) distribution.
    """
    p = 1 - q
    xs = [geometric(n, p) for i in range(2)]
    ys = xs[0] - xs[1]
    return ys


def uniform_double(n):
    """
    Samples a double-precision float uniformly from the interval [0,1).

    :param n: the number of samples to draw.
    :return: a 1D numpy array of length `n` of samples from the Uniform(0,1) distribution.
    """
    return backend.double_uniform(n)
