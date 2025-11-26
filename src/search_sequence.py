import numpy as np

def search_sequence_numpy(data, seq):
    """
    Recherche les positions où la sous-séquence seq apparaît dans data.
    Version 100% NumPy (rapide mais difficile à analyser pour Numba).
    """
    seq_ind = np.arange(seq.size)
    cor_size = data.size - seq.size + 1
    data_ind = np.arange(cor_size).reshape((cor_size, 1))
    
    return np.nonzero(np.all(data[data_ind + seq_ind] == seq, axis=1))[0]
