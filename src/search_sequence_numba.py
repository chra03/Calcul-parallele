import numpy as np
from numba import njit

@njit
def search_sequence_numba(data, seq):
    n = len(data)
    m = len(seq)
    out = []

    for i in range(n - m + 1):
        ok = True
        for j in range(m):
            if data[i + j] != seq[j]:
                ok = False
                break
        if ok:
            out.append(i)

    return out
