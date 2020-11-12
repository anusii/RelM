import numpy as np
from differential_privacy import backend


# def geometric(n, p=0.5):
#     """
#     Samples from the Geometric(p) distribution.
#
#     :param n: the number of samples to draw.
#     :param p: parameter that determines the spread of the distribution.
#     :return: a 1D numpy array of length `n` of samples from the Geometric(p) distribution.
#     """
#     # return backend.geometric(p, n).astype(np.int64)
#     # b = -1.0 / np.log(1.0 - p)
#     return backend.geometric(p, n)
#
#
# def uniform_double(n):
#     """
#     Samples a double-precision float uniformly from the interval [0,1).
#
#     :param n: the number of samples to draw.
#     :return: a 1D numpy array of length `n` of samples from the Uniform(0,1) distribution.
#     """
#     return backend.uniform_double(n)


def fixed_point_laplace(n, b=1, precision=35):
    """
    Samples from the Laplace(b) distribution.

    :param n: the number of samples to draw.
    :param b: parameter that determines the spread of the distribution.
    :return: a 1D numpy array of length `n` of samples from the Laplace(b) distribution.
    """
    return backend.fixed_point_laplace(b, n, precision)


# def fixed_point_exponential(n, b=1, precision=35):
#     """
#     Samples from the Exponential(b) distribution.
#
#     :param n: the number of samples to draw.
#     :param b: parameter that determines the spread of the distribution.
#     :return: a 1D numpy array of length `n` of samples from the Exponential(b) distribution.
#     """
#     return backend.fixed_point_exponential(b, n, precision)
