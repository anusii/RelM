import math
import numpy as np

from os import urandom
from numba import objmode, uint8, njit

BITS_PER_FLOAT = 53
BYTES_PER_FLOAT = math.ceil(BITS_PER_FLOAT / 8)
SHIFT_AMOUNT = 8*BYTES_PER_FLOAT - BITS_PER_FLOAT
NORMALIZER = 2**-BITS_PER_FLOAT

@njit
def _int_from_bytes(bs):
    """
    Converts bytes into an integer.

    :param bs: a bytes object to converted into an integer.
    :return: an big-endian integer representation of the input bytes.
    """
    accumulator = 0
    for i,b in enumerate(bs[::-1]):
        accumulator += b*(256)**i
    return accumulator

@njit
def _significands_from_bytes(bs, shift_amount = SHIFT_AMOUNT):
    n = len(bs) // BYTES_PER_FLOAT
    significands = np.zeros(n, dtype=np.int64)
    for i in range(n):
        start_index = BYTES_PER_FLOAT*i
        stop_index = BYTES_PER_FLOAT*(i+1)
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
    significands = np.zeros(n)
    significands = _significands_from_bytes(bs)
    unifs = significands * NORMALIZER
    return unifs

@njit
def uniform(n, a=0.0, b=1.0):
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
    unifs_01 = _floats_from_bytes(bs)
    unifs_ab = (b-a)*unifs_01 + a
    return unifs_ab

@njit
def exponential(n, b=1):
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

    :param n: the number of samples to draw.
    :param b: parameter that determines the spread of the distribution.
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
    # Notice that it is important that xs are chosen by uniform rather than by
    # uniform_double to prevent a circular dependency. Because the geometric
    # distribution is supported on the intervals it is not vulnerable to the
    # kinds of floating-point vulnerabilities that plague the Laplace
    # distribution and therefore the extra precision provided by uniform_double
    # is not required in this case.
    xs = uniform(n)
    # We want the geometric distribution to return an array of integers.
    # We use the math functions instead of the numpy functions here to produce
    # integer-type output that numba can understand.  Because this
    # function will be compiled in nopython mode these operations should not
    # slow things down too badly.
    ys = np.array([math.floor(math.log(x) / math.log(1-p)) for x in xs])
    return ys

@njit
def two_sided_geometric(n, q=0.5):
    """
    Samples from the TwoSidedGeometric(q) distribution.

    :param n: the number of samples to draw.
    :param q: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the TwoSidedGeometric(q) distribution.
    """
    # Notice that we use uniform instead of uniform_double to sample from the
    # Uniform(0,1) distribution in this function.  Because the TwoSidedGeometric
    # is supported on the integers we don't need the extra precision offered by
    # uniform_double.  So, we use uniform here because it is faster.
    xs = uniform(n, a=-0.5, b=0.5)
    xs *= (1 + q)
    sgn = np.sign(xs)
    # We want the two_sided_geometric distribution to return an array of integers.
    # We use the math functions instead of the numpy functions here to produce
    # integer-type output that numba can understand.  Because this
    # function will be compiled in nopython mode these operations should not
    # slow things down too badly.
    ys = np.array([int(sgn[i])*math.floor(math.log(sgn[i]*xs[i]) / math.log(q)) for i in range(n)])
    return ys

@njit
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

@njit
def simple_two_sided_geometric(n, q=0.5):
    """
    Samples from the TwoSidedGeometric(q) distribution.

    :param n: the number of samples to draw.
    :param q: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the TwoSidedGeometric(q) distribution.
    """
    p = 1-q
    xs = [geometric(n,p) for i in range(2)]
    ys = xs[0] - xs[1]
    return ys

@njit
def uniform_double(n):
    """
    Samples a double-precision float uniformly from the interval [0,1).

    :param n: the number of samples to draw.
    :return: a 1D numpy array of length `n` of samples from the Uniform(0,1) distribution.
    """
    bs = np.zeros(7*n, dtype=np.uint8)
    with objmode(bs='uint8[:]'):
        bs = np.frombuffer(urandom(7*n), dtype=np.uint8)
    significands = _significands_from_bytes(bs, shift_amount = (SHIFT_AMOUNT+1))
    # Add the implicit leading 1 to the significands.
    significands ^= 2**52
    # Generate the exponent for floats in the range [0,1].
    # Note that these exponents will take positive integer values.  While
    # this could technically overflow, the probability of that happening is
    # negligible (much less than 2**-512).
    exponents = geometric(n, p=0.5) + 1
    # Adjust the exponents to account for the fact that the significands
    # are represented here as ints rather than as floats in the interval [1,2)
    # as described in the IEEE standard.
    exponents += 52
    unifs = significands * (2.0 ** (-exponents))
    return unifs
