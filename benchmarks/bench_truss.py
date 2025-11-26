import time
import numpy as np
import sys, os

# === ajout crucial pour que Python trouve src/ ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.truss import truss
from src.truss_numba import truss_numba

def bench(fn, A, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(A)
        best = min(best, time.perf_counter() - t0)
    return best

if __name__ == "__main__":
    A = np.ones(10)

    # warm-up pour compiler numba
    truss_numba(A)

    t_py = bench(truss, A)
    t_nb = bench(truss_numba, A)

    print("Python :", t_py)
    print("Numba  :", t_nb)
    print("Speedup =", t_py / t_nb)
