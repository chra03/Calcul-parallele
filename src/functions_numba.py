import numpy as np
from numba import njit

@njit
def fast_sum_squares(arr):
    s = 0.0
    for i in range(arr.size):
        s += arr[i] * arr[i]
    return s
