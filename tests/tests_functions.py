import numpy as np
from functions import slow_sum_squares
from functions_numba import fast_sum_squares

def test_parity():
    arr = np.random.randn(10000).astype(np.float64)

    # warm-up : compile numba
    fast_sum_squares(arr)

    out_py = slow_sum_squares(arr)
    out_nb = fast_sum_squares(arr)

    assert np.allclose(out_py, out_nb)
