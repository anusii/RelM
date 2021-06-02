import numpy as np

def median_smooth_sensitivity_naive(x, epsilon, lower_bound, upper_bound):
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

def j_star_func(y, i, L, U):
    n = len(y) - 2
    m = (n+1)//2
    j = np.arange(L, U+1)
    return np.argmax((y[j] - y[i]) * np.exp(-epsilon * (j - (i+1)))) + L

def j_list_func(y, a, c, L, U):
    if c < a:
        return []
    else:
        b = (a+c)//2
        j_star_b = j_star_func(y, b, L, U)
        return j_list_func(y, a, b-1, L, j_star_b) + [j_star_b] + j_list_func(y, b+1, c, j_star_b, U)

def median_smooth_sensitivity(x, epsilon, lower_bound, upper_bound):
    n = len(x)
    m = (n+1)//2
    y = np.concatenate((np.array([lower_bound]), np.sort(x), np.array([upper_bound])))
    j = np.arange(m+1, n+2)
    j_star = j_star_func(y, m, m+1, n+1)
    ret = (y[j_star] - y[m]) * np.exp(-epsilon * (j_star - (m+1)))
    j_list = np.array(j_list_func(y, 0, m-1, m, n+1))
    i = np.arange(m, 0, -1)
    temp = (y[j_list] - y[m-i]) * np.exp(-epsilon * (j_list - m + (i-1)))
    return max(ret, np.max(temp))

def median_smooth_sensitivity_boolean(x, epsilon):
    n = len(x)
    m_low = n//2
    m_high = m_low + 1
    temp = min(np.abs(x.sum() - m_low), np.abs(x.sum() - m_high))
    return np.exp(-epsilon*temp)
