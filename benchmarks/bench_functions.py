import time
import numpy as np
from functions import slow_sum_squares
from functions_numba import fast_sum_squares

def bench(fn, arr, warmup=1, repeat=5):
    # warm up
    for _ in range(warmup):
        fn(arr)

    best = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(arr)
        best = min(best, time.perf_counter() - t0)
    return best

if __name__ == "__main__":

    arr = np.random.randn(2_000_000).astype(np.float64)

    t_py = bench(slow_sum_squares, arr)
    t_nb = bench(fast_sum_squares, arr)

    print(f"Python  : {t_py:.5f} sec")
    print(f"Numba   : {t_nb:.5f} sec")
    print(f"Speedup = x{t_py/t_nb:.1f}")
