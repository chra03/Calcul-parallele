import numpy as np
from numba import njit, prange

@njit(parallel=True)
def search_sequence_numba_parallel(data, seq):
    """
    Version parallèle de search_sequence.
    Utilise prange pour paralléliser la boucle externe.
    """
    n = len(data)
    m = len(seq)
    matches = np.zeros(n - m + 1, dtype=np.uint8)

    for i in prange(n - m + 1):  # boucle parallèle
        ok = True
        for j in range(m):
            if data[i + j] != seq[j]:
                ok = False
                break
        if ok:
            matches[i] = 1

    return list(np.nonzero(matches)[0])
