import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search_sequence import search_sequence_numpy
from src.search_sequence_simple import search_sequence_python
from src.search_sequence_numba import search_sequence_numba

def test_parity():
    data = np.random.randint(0, 10, size=1000).astype(np.uint8)
    seq  = np.array([3, 5], dtype=np.uint8)

    # warm-up pour compiler Numba
    search_sequence_numba(data, seq)

    out_np = search_sequence_numpy(data, seq)
    out_py = search_sequence_python(data, seq)
    out_nb = search_sequence_numba(data, seq)

    assert list(out_np) == list(out_py)
    assert list(out_nb) == list(out_py)
print("Parité OK ✔️")
