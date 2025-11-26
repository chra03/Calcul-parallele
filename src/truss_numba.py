import numpy as np
from numba import njit
from math import sin, cos

# -----------------------------
# Version Numba de bar()
# -----------------------------
@njit
def bar_numba(E, A, L, phi):
    c = np.cos(phi)
    s = np.sin(phi)

    k0 = np.array([[c*c, c*s], [c*s, s*s]])
    k1 = np.hstack((k0, -k0))
    K = (E*A/L) * np.vstack((k1, -k1))

    S = (E/L) * np.array([-c, -s, c, s])

    return K, S


# -----------------------------
# Version Numba de node2idx()
# -----------------------------
@njit
def node2idx_numba(node, DOF):
    idx = np.empty(len(node) * DOF, dtype=np.int64)
    k = 0
    for n in node:
        start = DOF * (n - 1)
        finish = DOF * n
        for i in range(start, finish):
            idx[k] = i
            k += 1
    return idx


# -----------------------------
# Version Numba partielle de truss()
# -----------------------------
@njit
def truss_numba(A):
    P = 1e5
    Ls = 360.0
    Ld = np.sqrt(360**2 * 2)

    start = np.array([5, 3, 6, 4, 4, 2, 5, 6, 3, 4])
    finish = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1])
    phi = np.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45]) * np.pi / 180
    L = np.array([Ls, Ls, Ls, Ls, Ls, Ls, Ld, Ld, Ld, Ld])

    nbar = len(A)
    E = 1e7 * np.ones(nbar)
    rho = 0.1 * np.ones(nbar)

    Fx = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Fy = np.array([0.0, -P, 0.0, -P, 0.0, 0.0])
    rigid = np.array([False, False, False, False, True, True])

    n = len(Fx)
    DOF = 2

    mass = np.sum(rho * A * L)

    K = np.zeros((DOF*n, DOF*n))
    S = np.zeros((nbar, DOF*n))

    # HOTSPOT principal optimisé
    for i in range(nbar):
        Ksub, Ssub = bar_numba(E[i], A[i], L[i], phi[i])
        idx = node2idx_numba(np.array([start[i], finish[i]]), DOF)
        for ii in range(4):
            for jj in range(4):
                K[idx[ii], idx[jj]] += Ksub[ii, jj]
        S[i, idx] = Ssub

    return mass, S
