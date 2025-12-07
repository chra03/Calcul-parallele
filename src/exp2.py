import numpy as np


def accumulate_matrix(A, B):
    # Hotspot : double boucle
    m, n = A.shape
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            C[i, j] = A[i, j] + B[i, j] * 2.0
    return C


def sum_rows(X):
    # Hotspot : boucle simple
    n, d = X.shape
    out = np.zeros(n)
    for i in range(n):
        total = 0.0
        for j in range(d):
            total += X[i, j]
        out[i] = total
    return out


def normalize_signal(x):
    # Hotspot : boucle avec opérations math
    for i in range(len(x)):
        x[i] = (x[i] - 0.5) * 2.0
    return x


def constant_function(k):
    # PAS hotspot (aucune boucle)
    return k * 3.14
