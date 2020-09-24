import numpy as np

from os import urandom
from numba import objmode, uint8, njit

BPF = 53
RECIP_BPF = 2**-BPF

@njit
def _int_from_bytes(bs):
    accum = 0
    for i,b in enumerate(bs[::-1]):
        accum += b*(256)**i
    return accum

@njit
def _unifs_from_bytes(bs, n):
    unifs = np.zeros(n)
    for i in range(n):
        start_index = 7*i
        stop_index = 7*(i+1)
        unifs[i] = _int_from_bytes(bs[start_index:stop_index]) >> 3
    unifs *= RECIP_BPF
    return unifs

@njit
def uniform(n, a=0, b=1):
    """
    Samples from the uniform distribution on the interval [a,b).

    :param n: the number of samples to draw.
    :param a: the lower bound of the interval to draw from.
    :param b: the upper bound of the interval to draw from.
    :return: a 1D numpy array of length `n` of samples from the Uniform(0,1) distribution.
    """
    bs = np.zeros(7*n, dtype=np.uint8)
    with objmode(bs='uint8[:]'):
        bs = np.frombuffer(urandom(7*n), dtype=np.uint8)
    unifs_01 = _unifs_from_bytes(bs, n)
    unifs_ab = (b-a)*unifs_01 + a
    return unifs_ab

@njit
def exponential(n, b):
    """
    Samples from the Exponential(b) distribution.

    :param n: the number of samples to draw.
    :param b: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the Exponential(b) distribution.
    """
    xs = uniform(n)
    ys = -b*np.log(xs)
    return ys


@njit
def laplace(n, b=1):
    """
    Samples from the Laplace(b) distribution.

    :param b: parameter that determines the spread of the distribution.
    :param n: the number of samples to draw.
    :return: a 1D numpy array of length `n` of samples from the Laplace(b) distribution.
    """
    xs = uniform(n, a=-0.5, b=0.5)
    sgn = np.sign(xs)
    ys = sgn * np.log(2 * sgn * xs) * b
    return ys

@njit
def geometric(n, p=0.5):
    """
    Samples from the Geometric(p) distribution.

    :param n: the number of samples to draw.
    :param p: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the Geometric(p) distribution.
    """
    xs = uniform(n)
    ys = np.floor(np.log(xs) / np.log(1-p))
    return ys

@njit
def two_sided_geometric(n, q=0.5):
    """
    Samples from the TwoSidedGeometric(q) distribution.

    :param q: parameter that determines the spread of the distribution.
    :param n: the number of samples to draw.
    :return: a 1D numpy array of length `n` of samples from the TwoSidedGeometric(q) distribution.
    """
    xs = uniform(n, a=-0.5, b=0.5)
    xs *= (1 + q)
    sgn = np.sign(xs)
    ys = sgn * np.floor(np.log(sgn * xs) / np.log(q))
    return ys

@njit
def simple_laplace(n, b=1):
    xs = [exponential(n, b) for i in range(2)]
    ys = xs[0] - xs[1]
    return ys

@njit
def simple_two_sided_geometric(n, q=0.5):
    p = 1-q
    xs = [geometric(n,p) for i in range(2)]
    ys = xs[0] - xs[1]
    return ys
