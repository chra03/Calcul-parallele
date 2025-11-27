import numpy as np
import sys, os

# ajoute le dossier racine au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.truss import truss
from src.truss_numba import truss_numba

def test_truss_parity():
    A = np.ones(10)

    # warm-up Numba
    truss_numba(A)

    mass_py, stress_py = truss(A)
    mass_nb, stress_nb = truss_numba(A)

    assert abs(mass_py - mass_nb) < 1e-6
    assert np.allclose(stress_py, stress_nb, atol=1e-6)

print("Parité truss OK ✔️")