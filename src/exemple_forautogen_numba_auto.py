from numba import njit
import numpy as np

import numpy as np


def normalize(x):
    # Fonction simple : PAS un hotspot
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def add_vectors(a, b):
    # Fonction simple : pas de boucle lente
    return a + b


@njit
def naive_convolution(signal, kernel):
    """
    Fonction hotspot : version 100% Python d'une convolution.
    Hyper lente pour les grands signaux.
    C’est celle que ton auto-optimiseur doit repérer.
    """
    n = len(signal)
    k = len(kernel)
    out = np.zeros(n - k + 1)

    for i in range(n - k):
        s = 0.0
        for j in range(k):
            s += signal[i + j] * kernel[j]
        out[i] = s

    return out


def say_hello():
    # Fonction totalement inutile, juste pour tester
    return "Hello"
