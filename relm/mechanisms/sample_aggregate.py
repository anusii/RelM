import numpy as np

def median_smooth_sensitivity_naive(x, epsilon, lower_bound, upper_bound):
    """
    Compute the smooth sensitivity of the median function.
    This is a naive implementation that is easy to understand but
    much slower than the improved algorithm implemented by the function
    median_smooth_sensitivity() below.

    Args:
        x: a vector of floating point numbers
        epsilon: the maximum privacy loss of the mechanism
        lower_bound: the smallest possible value for any component of x
        upper_bound: the largest possible value for any compnent of x
    """

    n = len(x)
    m = (n+1)//2
    y = np.concatenate((np.array([lower_bound]), np.sort(x), np.array([upper_bound])))
    ret = 0
    for k in range(n+1):
        t = np.arange(k+2)
        high = np.minimum(m + t, n+1)
        low = np.maximum(m+t-(k+1), 0)
        y_diff = y[high] - y[low]
        temp = np.max(np.exp(-k*epsilon) * y_diff)
        if temp > ret:
            ret = temp
    return ret

def j_star_func(y, i, L, U, epsilon):
    n = len(y) - 2
    m = (n+1)//2
    j = np.arange(L, U+1)
    return np.argmax((y[j] - y[i]) * np.exp(-epsilon * (j - (i+1)))) + L

def j_list_func(y, a, c, L, U, epsilon):
    if c < a:
        return []
    else:
        b = (a+c)//2
        j_star_b = j_star_func(y, b, L, U, epsilon)
        first_half = j_list_func(y, a, b-1, L, j_star_b, epsilon)
        second_half = j_list_func(y, b+1, c, j_star_b, U, epsilon)
        return first_half + [j_star_b] + second_half

def median_smooth_sensitivity(x, epsilon, lower_bound, upper_bound):
    """
    Compute the smooth sensitivity of the median function.
    This is an efficient implementation that uses Algorithm 1 given
    in "Smooth Sensitivity and Sampling in Private Data Analysis" to
    compute the sensitivity exactly in time O(n log n).


    Args:
        x: a vector of floating point numbers
        epsilon: the maximum privacy loss of the mechanism
        lower_bound: the smallest possible value for any component of x
        upper_bound: the largest possible value for any compnent of x
    """
    n = len(x)
    m = (n+1)//2
    y = np.concatenate((np.array([lower_bound]), np.sort(x), np.array([upper_bound])))
    j = np.arange(m+1, n+2)
    j_star = j_star_func(y, m, m+1, n+1, epsilon)
    ret = (y[j_star] - y[m]) * np.exp(-epsilon * (j_star - (m+1)))
    j_list = np.array(j_list_func(y, 0, m-1, m, n+1, epsilon))
    i = np.arange(m, 0, -1)
    temp = (y[j_list] - y[m-i]) * np.exp(-epsilon * (j_list - m + (i-1)))
    return max(ret, np.max(temp))

def median_smooth_sensitivity_boolean(x, epsilon):
    """
    Compute the smooth sensitivity of the median on boolean inputs.
    This is much faster than the algorithm that computes the smooth
    sensitivity for the median on floating point inputs.

    Args:
        x: a vector of booleans
        epsilon: the maximum privacy loss of the mechanism
    """
    n = len(x)
    m_low = n//2
    m_high = m_low + 1
    temp = min(np.abs(x.sum() - m_low), np.abs(x.sum() - m_high))
    return np.exp(-epsilon*temp)


class SampleAggregate(ReleaseMechanism):
    def __init__(self, epsilon, lower_bound, upper_bound, method="median"):
        super(SampleAggregate, self).__init__(epsilon)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def release(self, values):
        self._check_valid()
        self._is_valid = False
        self._update_accountant()

        smooth_sensitivity = compute_smooth_sensitivity(values,
                                                        self.epsilon,
                                                        self.lower_bound,
                                                        self.upper_bound)
        beta = self.epsilon / 6.0
        mechanism = CauchyMechanism(epsilon, beta)
        query_response = mechanism.release(np.median(values), smooth_sensitivity)
        return query_response

    @property
    def privacy_consumed(self):
        """
        Computes the privacy budget consumed by the mechanism so far.
        """
        if self._is_valid:
            return 0
        else:
            return self.epsilon
