import time
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.search_sequence_simple import search_sequence_python
from src.search_sequence_numba import search_sequence_numba

def bench(fn, data, seq, warmup=1, repeat=5):
    for _ in range(warmup):
        fn(data, seq)

    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(data, seq)
        best = min(best, time.perf_counter() - t0)
    return best

if __name__ == "__main__":
    data = np.random.randint(0, 10, size=1_000_000).astype(np.uint8)
    seq  = np.array([3, 5], dtype=np.uint8)

    t_py = bench(search_sequence_python, data, seq)
    t_nb = bench(search_sequence_numba, data, seq)

    print(f"Python : {t_py:.5f} sec")
    print(f"Numba  : {t_nb:.5f} sec")
    print(f"Speedup = ×{t_py/t_nb:.1f}")
