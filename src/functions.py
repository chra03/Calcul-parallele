import numpy as np

def slow_sum_squares(arr):
    """
    Version Python lente (boucles natives)
    """
    s = 0.0
    for x in arr:
        s += x * x
    return s
