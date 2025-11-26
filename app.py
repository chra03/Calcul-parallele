import streamlit as st
import numpy as np
import time
import inspect
import pandas as pd
import cProfile, pstats
import io
from pathlib import Path

# ----------------------------
# Import local modules
# ----------------------------
from src.search_sequence_simple import search_sequence_python
from src.search_sequence_numba import search_sequence_numba
from src.search_sequence_numba_parallel import search_sequence_numba_parallel

from src.truss import truss
from src.truss_numba import truss_numba

# For AST analysis
from analysis.static_analysis import analyze_file
from analysis.static_analysis_truss import analyze_file_truss


# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Projet Numba – Calcul parallèle",
    layout="wide"
)

st.title("Projet Numba – Calcul parallèle")
st.write("""
Refactorisation de fonctions Python coûteuses avec Numba,  
selon une méthodologie complète : analyse statique, profiling, optimisation,  
tests de parité et benchmarks.
""")

# ======================================================================
# TABS
# ======================================================================
tabs = st.tabs([
    "📘 Sum of squares",
    "🔍 Search Sequence",
    "🏗️ Truss 10 barres",
    "📊 Analyse statique & Profiling",
    "🧠 Synthèse globale"
])

# ======================================================================
# 1) SUM OF SQUARES
# ======================================================================
with tabs[0]:
    st.header("Exemple simple : somme des carrés")

    # Python version
    def sum_squares_py(n):
        s = 0
        for i in range(n):
            s += i * i
        return s

    # Numba version
    from numba import njit
    @njit
    def sum_squares_nb(n):
        s = 0
        for i in range(n):
            s += i * i
        return s

    n = st.slider("Taille du tableau", 10_000, 2_000_000, 100_000)

    if st.button("Bench sum of squares"):
        # Warm-up
        sum_squares_nb(10)

        # Bench
        t0 = time.perf_counter()
        sum_squares_py(n)
        t_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        sum_squares_nb(n)
        t_nb = time.perf_counter() - t0

        st.success(f"Python : {t_py:.6f}s — Numba : {t_nb:.6f}s — Speedup ×{t_py/t_nb:.1f}")

        # Chart
        df = pd.DataFrame({
            "Version": ["Python", "Numba"],
            "Temps (s)": [t_py, t_nb]
        })
        st.bar_chart(df, x="Version", y="Temps (s)")


# ======================================================================
# 2) SEARCH SEQUENCE
# ======================================================================
with tabs[1]:
    st.header("Cas 1 : recherche de sous-séquence")

    size = st.slider("Taille de data", 50_000, 2_000_000, 500_000)

    if st.button("Bench search sequence"):
        data = np.random.randint(0, 10, size).astype(np.uint8)
        seq = np.array([3, 5], dtype=np.uint8)

        # Warm-up
        search_sequence_numba(data, seq)
        search_sequence_numba_parallel(data, seq)

        # Bench
        def bench(fn):
            t0 = time.perf_counter()
            fn(data, seq)
            return time.perf_counter() - t0

        t_py = bench(search_sequence_python)
        t_nb = bench(search_sequence_numba)
        t_nb_p = bench(search_sequence_numba_parallel)

        st.success(f"""
        Python : {t_py:.6f}s  
        Numba : {t_nb:.6f}s  
        Numba Parallel : {t_nb_p:.6f}s  
        Speedup Python→Numba = ×{t_py/t_nb:.1f}  
        """)

        df = pd.DataFrame({
            "Version": ["Python", "Numba", "Numba Parallel"],
            "Temps (s)": [t_py, t_nb, t_nb_p]
        })
        st.bar_chart(df, x="Version", y="Temps (s)")

    if st.button("Tester la parité (search sequence)"):
        data = np.random.randint(0, 10, 2000).astype(np.uint8)
        seq = np.array([3, 5], dtype=np.uint8)

        # Compile Numba
        search_sequence_numba(data, seq)

        py = search_sequence_python(data, seq)
        nb = search_sequence_numba(data, seq)

        if py == nb:
            st.success("Parité OK.")
        else:
            st.error("Parité NON vérifiée.")


# ======================================================================
# 3) TRUSS
# ======================================================================
with tabs[2]:
    st.header("Cas 2 : structure en treillis (10-bar Truss)")

    A_scale = st.slider("Facteur sur les sections A", 0.1, 5.0, 1.0)

    # Codes ESSENTIELS
    code_truss_python = """
# ➤ Hotspot Python (assemblage global)

for i in range(nbar):
    Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])
    idx = node2idx([start[i], finish[i]], DOF)
    K[np.ix_(idx, idx)] += Ksub
    S[i, idx] = Ssub
"""

    code_truss_numba = """
# ➤ Hotspot optimisé (Numba)

for i in range(nbar):
    Ksub, Ssub = bar_numba(E[i], A[i], L[i], phi[i])
    idx = node2idx_numba(np.array([start[i], finish[i]]), DOF)
    for ii in range(4):
        for jj in range(4):
            K[idx[ii], idx[jj]] += Ksub[ii, jj]
    S[i, idx] = Ssub
"""

    with st.expander("🔥 Code hotspot Python original"):
        st.code(code_truss_python, language="python")

    with st.expander("⚡ Code hotspot optimisé (Numba)"):
        st.code(code_truss_numba, language="python")

    # Bench
    if st.button("Bench Truss"):
        A = A_scale * np.ones(10)

        # Warm-up
        truss_numba(A)

        t0 = time.perf_counter()
        truss(A)
        t_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        truss_numba(A)
        t_nb = time.perf_counter() - t0

        st.success(f"Python : {t_py:.6f}s — Numba : {t_nb:.6f}s — Speedup ×{t_py/t_nb:.1f}")

        df = pd.DataFrame({
            "Version": ["Python", "Numba"],
            "Temps (s)": [t_py, t_nb]
        })
        st.bar_chart(df, x="Version", y="Temps (s)")

    # Parité
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Tester la parité (truss)"):
            A = np.ones(10)

            mass_py, stress_py = truss(A)
            mass_nb, stress_nb = truss_numba(A)

            same_mass = abs(mass_py - mass_nb) < 1e-6
            same_stress = np.allclose(
                np.array(stress_py).flatten(),
                np.array(stress_nb).flatten(),
                atol=1e-6
            )

            if same_mass and same_stress:
                st.success("Parité OK : masses + contraintes identiques.")
            else:
                st.error("Parité NON vérifiée !")


# ======================================================================
# 4) ANALYSE STATIQUE + PROFILING
# ======================================================================
with tabs[3]:
    st.header("Analyse statique (AST)")

    files = [
        "src/search_sequence_simple.py",
        "src/search_sequence_numba_parallel.py",
        "src/truss.py"
    ]

    rows = []
    for f in files:
        stats = analyze_file(f)
        rows.append([f, stats["lines"], stats["fors"], stats["ifs"], stats["calls"]])

    df = pd.DataFrame(rows, columns=["Fichier", "Lignes", "Boucles for", "If", "Appels"])
    st.dataframe(df)

    st.subheader("Profiling cProfile (search_sequence_python)")

    if st.button("Profiler search_sequence_python"):
        data = np.random.randint(0, 10, 3000).astype(np.uint8)
        seq = np.array([3, 5], dtype=np.uint8)

        pr = cProfile.Profile()
        pr.enable()
        search_sequence_python(data, seq)
        pr.disable()

        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats('cumtime').print_stats(15)
        st.text(s.getvalue())


# ======================================================================
# 5) SYNTHÈSE
# ======================================================================
with tabs[4]:
    st.header("Synthèse générale")

    st.write("""
### 🧠 Pipeline général appliqué :
1. Analyse statique (AST)
2. Profiling cProfile
3. Optimisation Numba
4. Tests de parité
5. Benchmarks
6. Conclusion et reproductibilité
""")

    st.success("Tout le pipeline a été appliqué sur 3 cas : Sum of squares, Search Sequence et Truss.")


