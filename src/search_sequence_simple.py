def search_sequence_python(data, seq):
    """
    Version Python de base.
    Boucles explicites (idéales pour Numba).
    """
    n = len(data)
    m = len(seq)
    matches = []

    for i in range(n - m + 1):
        ok = True
        for j in range(m):
            if data[i + j] != seq[j]:
                ok = False
                break
        if ok:
            matches.append(i)

    return matches
