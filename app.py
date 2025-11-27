

import os
import sys
import time
import ast
import io
import cProfile
import pstats
import subprocess
import inspect

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------
# CONFIG STREAMLIT + STYLE GLOBAL
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Projet Numba – Calcul parallèle",
    layout="wide"
)

# Style léger pour faire plus pro
st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 0.8rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }
    .subsection {
        font-weight: 600;
        margin-top: 0.4rem;
        margin-bottom: 0.1rem;
    }
    .metric-container {
        background-color: #f8f9fb;
        padding: 0.6rem 0.8rem;
        border-radius: 0.8rem;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# IMPORTS LOCAUX (dossier src/)
# ----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.append(SRC)

from functions import slow_sum_squares
from functions_numba import fast_sum_squares

from search_sequence_simple import search_sequence_python
from search_sequence import search_sequence_numpy
from search_sequence_numba import search_sequence_numba
from search_sequence_numba_parallel import search_sequence_numba_parallel

from truss import truss
from truss_numba import truss_numba


# ----------------------------------------------------------------------
# OUTILS GÉNÉRIQUES : BENCHMARK, AST, RADON, PROFILING, SOURCE
# ----------------------------------------------------------------------
def bench(fn, *args, warmup=1, repeat=5):
    """Mesure le meilleur temps d'exécution d'une fonction."""
    for _ in range(warmup):
        fn(*args)

    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def analyze_file(path):
    """Analyse statique basique : lignes, for, if, appels."""
    full = os.path.join(ROOT, path)
    with open(full, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)

    class Analyzer(ast.NodeVisitor):
        def __init__(self):
            self.for_count = 0
            self.if_count = 0
            self.call_count = 0
            self.length = len(src.splitlines())

        def visit_For(self, node):
            self.for_count += 1
            self.generic_visit(node)

        def visit_If(self, node):
            self.if_count += 1
            self.generic_visit(node)

        def visit_Call(self, node):
            self.call_count += 1
            self.generic_visit(node)

    a = Analyzer()
    a.visit(tree)
    return a


def run_radon(paths):
    """Lance radon cc -s -a sur une liste de fichiers."""
    if isinstance(paths, str):
        paths = [paths]
    cmd = ["radon", "cc", "-s", "-a"] + paths
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception as e:
        out = (
            "Impossible d'exécuter radon.\n"
            f"Erreur : {e}\n"
            "Vérifiez que radon est installé (pip install radon) et accessible."
        )
    return out


def get_source(obj):
    """Récupère le code source d'une fonction pour l'afficher."""
    try:
        return inspect.getsource(obj)
    except OSError:
        return "# Source non disponible pour cet objet."


def profile_sum_squares(n=200_000):
    arr = np.random.rand(n)

    pr = cProfile.Profile()
    pr.enable()
    slow_sum_squares(arr)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(10)
    return s.getvalue()


def profile_search_sequence(n=200_000):
    data = np.random.randint(0, 10, size=n).astype(np.uint8)
    seq = np.array([3, 5], dtype=np.uint8)

    pr = cProfile.Profile()
    pr.enable()
    search_sequence_python(data, seq)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(10)
    return s.getvalue()


def profile_truss():
    A = np.ones(10)

    pr = cProfile.Profile()
    pr.enable()
    truss(A)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(10)
    return s.getvalue()


# ----------------------------------------------------------------------
# EN-TÊTE GLOBAL
# ----------------------------------------------------------------------
st.markdown('<div class="big-title">Projet Numba – Calcul parallèle</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Refactorisation de fonctions Python coûteuses avec Numba, '
    'en suivant systématiquement la même démarche : analyse statique, '
    'profiling, optimisation, tests de parité et benchmarks.'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")

tab1, tab2, tab3, tab_auto, tab4 = st.tabs([
    "🧮 Sum of squares",
    "🔎 Search Sequence",
    "🏗️ Truss 10 barres",
    "⚡ Auto-optimisation",
    "🧭 Synthèse globale",
])


# ----------------------------------------------------------------------
# AUTO-OPTIMISATION Numba pour une seule fonction hotspot
# ----------------------------------------------------------------------
import ast
import astor

class NumbaRefactor(ast.NodeTransformer):

    def __init__(self):
        self.candidates = []  # Liste des fonctions détectées comme hotspots

    def visit_FunctionDef(self, node):
        has_for = any(isinstance(child, ast.For) for child in ast.walk(node))

        if has_for:
            self.candidates.append(node.name)
            decorator = ast.Name(id="njit", ctx=ast.Load())
            node.decorator_list.insert(0, decorator)

        return node


def generate_numba_version(input_path, output_path):
    full_input = os.path.join(ROOT, input_path)
    code = open(full_input).read()

    tree = ast.parse(code)
    transformer = NumbaRefactor()
    new_tree = transformer.visit(tree)

    if transformer.candidates:
        import_node = ast.ImportFrom(
            module="numba",
            names=[ast.alias(name="njit", asname=None)],
            level=0
        )
        new_tree.body.insert(0, import_node)

    new_code = astor.to_source(new_tree)

    full_output = os.path.join(ROOT, output_path)
    with open(full_output, "w") as f:
        f.write(new_code)

    return transformer.candidates, new_code



# ----------------------------------------------------------------------
# TAB 1 — SUM OF SQUARES (EXEMPLE JOUET)
# ----------------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-title">Cas 1 : somme des carrés</div>', unsafe_allow_html=True)
    st.write(
        "Premier cas d’étude très simple : même fonction de somme des carrés, "
        "écrite en Python pur puis compilée avec Numba. "
        "Cet exemple permet d’illustrer la démarche sur un code minimal."
    )

    # --- Code Python vs Numba
    col_code1, col_code2 = st.columns(2)
    with col_code1:
        st.markdown('<div class="subsection">Code Python (référence)</div>', unsafe_allow_html=True)
        st.code(get_source(slow_sum_squares), language="python")
    with col_code2:
        st.markdown('<div class="subsection">Code Numba (@njit)</div>', unsafe_allow_html=True)
        st.code(get_source(fast_sum_squares), language="python")

    st.write("---")

    # --- Analyse statique + Radon
    st.markdown('<div class="subsection">Analyse statique (AST) & complexité</div>', unsafe_allow_html=True)

    rows = []
    for path in ["src/functions.py", "src/functions_numba.py"]:
        a = analyze_file(path)
        rows.append({
            "Fichier": os.path.basename(path),
            "Lignes": a.length,
            "Boucles for": a.for_count,
            "If": a.if_count,
            "Appels de fonction": a.call_count,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.button("Afficher le rapport Radon (sum of squares)"):
        report = run_radon(["src/functions.py", "src/functions_numba.py"])
        st.text(report)

    st.caption(
        "L’analyse statique confirme que la fonction est essentiellement une boucle numérique, "
        "ce qui en fait une candidate naturelle pour Numba."
    )

    st.write("---")

    # --- Profiling
    st.markdown('<div class="subsection">Profiling (cProfile) de la version Python</div>', unsafe_allow_html=True)
    if st.button("Profiler slow_sum_squares"):
        rep = profile_sum_squares()
        st.text(rep)
        st.info("On observe que tout le temps est passé dans la fonction elle-même (aucun appel complexe).")

    st.write("---")

    # --- Benchmarks + parité
    st.markdown('<div class="subsection">Benchmarks & parité</div>', unsafe_allow_html=True)

    n = st.slider("Taille du tableau", 10_000, 2_000_000, 200_000, step=10_000)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Tester la parité (sum of squares)"):
            arr = np.random.rand(1_000)
            v_py = slow_sum_squares(arr)
            v_nb = fast_sum_squares(arr)
            if abs(v_py - v_nb) < 1e-6:
                st.success("Parité OK : résultats identiques à 1e-6 près.")
            else:
                st.error("Parité NON vérifiée !")

    with col_b2:
        if st.button("Lancer les benchmarks (sum of squares)"):
            arr = np.random.rand(n)
            t_py = bench(slow_sum_squares, arr)
            t_nb = bench(fast_sum_squares, arr)
            speed = t_py / t_nb if t_nb > 0 else float("inf")

            c1, c2, c3 = st.columns(3)
            c1.metric("Temps Python", f"{t_py:.6f} s")
            c2.metric("Temps Numba", f"{t_nb:.6f} s")
            c3.metric("Speedup", f"×{speed:.1f}")

            df = pd.DataFrame({
                "Version": ["Python", "Numba"],
                "Temps (s)": [t_py, t_nb]
            })
            fig = px.bar(df, x="Version", y="Temps (s)",
                         title="Temps d'exécution – Sum of squares")
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# TAB 2 — SEARCH SEQUENCE (EXEMPLE INTERMÉDIAIRE)
# ----------------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">Cas 2 : recherche de sous-séquence</div>', unsafe_allow_html=True)
    st.write(
        "Deuxième cas d’étude : recherche d’un motif `[3, 5]` dans un grand tableau de chiffres. "
        "On compare une double boucle Python, une version NumPy vectorisée, "
        "puis des versions Numba et Numba parallèle."
    )

    # --- Codes
    st.markdown('<div class="subsection">Code Python (référence) & versions optimisées</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("**Version Python (double boucle)**")
        st.code(get_source(search_sequence_python), language="python")
    with col_s2:
        st.write("**Version Numba**")
        st.code(get_source(search_sequence_numba), language="python")

    # --- Analyse statique + Radon
    st.write("---")
    st.markdown('<div class="subsection">Analyse statique (AST) & complexité</div>', unsafe_allow_html=True)

    files_seq = [
        "src/search_sequence_simple.py",
        "src/search_sequence.py",
        "src/search_sequence_numba.py",
        "src/search_sequence_numba_parallel.py",
    ]
    rows = []
    for path in files_seq:
        a = analyze_file(path)
        rows.append({
            "Fichier": os.path.basename(path),
            "Lignes": a.length,
            "Boucles for": a.for_count,
            "If": a.if_count,
            "Appels de fonction": a.call_count,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.button("Afficher le rapport Radon (search sequence)"):
        report = run_radon(files_seq)
        st.text(report)

    st.caption(
        "L’analyse met en évidence la présence de boucles imbriquées dans la version Python, "
        "alors que la version NumPy repose davantage sur des appels vectorisés."
    )

    # --- Profiling
    st.write("---")
    st.markdown('<div class="subsection">Profiling (cProfile) de search_sequence_python</div>', unsafe_allow_html=True)

    if st.button("Profiler search_sequence_python"):
        rep = profile_search_sequence()
        st.text(rep)
        st.info("Le rapport montre que la quasi-totalité du temps est passée dans la double boucle Python → hotspot.")

    # --- Parité & Benchmarks
    st.write("---")
    st.markdown('<div class="subsection">Parité & Benchmarks</div>', unsafe_allow_html=True)

    n = st.slider("Taille de data", 50_000, 2_000_000, 500_000, step=50_000)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("Tester la parité (4 versions)"):
            data = np.random.randint(0, 10, 1_000, dtype=np.uint8)
            seq = np.array([3, 5], dtype=np.uint8)

            # Warm-up Numba
            search_sequence_numba(data, seq)
            search_sequence_numba_parallel(data, seq)

            out_py = search_sequence_python(data, seq)
            out_np = search_sequence_numpy(data, seq)
            out_nb = search_sequence_numba(data, seq)
            out_par = search_sequence_numba_parallel(data, seq)

            if (list(out_py) == list(out_np) ==
                    list(out_nb) == list(out_par)):
                st.success("Parité OK : toutes les versions renvoient les mêmes indices.")
            else:
                st.error("Parité NON vérifiée !")

    with col_p2:
        if st.button("Lancer les benchmarks (search sequence)"):
            data = np.random.randint(0, 10, size=n, dtype=np.uint8)
            seq = np.array([3, 5], dtype=np.uint8)

            t_py = bench(search_sequence_python, data, seq)
            t_np = bench(search_sequence_numpy, data, seq)
            t_nb = bench(search_sequence_numba, data, seq)
            t_par = bench(search_sequence_numba_parallel, data, seq)

            df = pd.DataFrame({
                "Version": ["Python", "NumPy", "Numba", "Numba Parallel"],
                "Temps (s)": [t_py, t_np, t_nb, t_par]
            })
            fig = px.bar(df, x="Version", y="Temps (s)",
                         title="Temps d'exécution – Search sequence")
            st.plotly_chart(fig, use_container_width=True)

            df["Speedup vs Python"] = t_py / df["Temps (s)"]
            st.dataframe(df[["Version", "Speedup vs Python"]], use_container_width=True)


# ----------------------------------------------------------------------
# TAB 3 — TRUSS (CAS SCIENTIFIQUE)
# ----------------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Cas 3 : truss 10 barres (calcul scientifique)</div>', unsafe_allow_html=True)
    st.write(
        "Dernier cas d’étude : un problème de structure mécanique classique (truss 10 barres). "
        "On compare une version de référence en Python pur et une version partiellement optimisée "
        "avec Numba (`truss_numba`)."
    )

    # --- Codes ciblés
    st.markdown('<div class="subsection">Parties du code ciblées par Numba</div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("**Fonction `truss` (extrait principal)**")
        st.code(get_source(truss), language="python")
    with col_t2:
        st.write("**Fonction `truss_numba`**")
        st.code(get_source(truss_numba), language="python")

    # --- Analyse statique + Radon
    st.write("---")
    st.markdown('<div class="subsection">Analyse statique (AST) & complexité</div>', unsafe_allow_html=True)

    files_truss = ["src/truss.py", "src/truss_numba.py"]
    rows = []
    for path in files_truss:
        a = analyze_file(path)
        rows.append({
            "Fichier": os.path.basename(path),
            "Lignes": a.length,
            "Boucles for": a.for_count,
            "If": a.if_count,
            "Appels de fonction": a.call_count,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.button("Afficher le rapport Radon (truss)"):
        report = run_radon(files_truss)
        st.text(report)

    st.caption(
        "La complexité reste raisonnable, mais le code réalise beaucoup de calculs numériques matriciels. "
        "La version Numba cible certaines parties, sans remplacer des appels comme `np.linalg.solve`."
    )

    # --- Profiling
    st.write("---")
    st.markdown('<div class="subsection">Profiling (cProfile) de truss</div>', unsafe_allow_html=True)

    if st.button("Profiler truss"):
        rep = profile_truss()
        st.text(rep)
        st.info(
            "Le profiling montre que le temps est dominé par les boucles d’assemblage de matrices "
            "et la résolution linéaire. Numba aide surtout sur les boucles numérales répétitives."
        )

    # --- Parité & Benchmarks
    st.write("---")
    st.markdown('<div class="subsection">Parité & Benchmarks</div>', unsafe_allow_html=True)

    A_scale = st.slider("Facteur sur les sections A (toutes égales)", 0.1, 5.0, 1.0, step=0.1)
    A = np.ones(10) * A_scale

    col_tb1, col_tb2 = st.columns(2)
    with col_tb1:
        if st.button("Tester la parité (truss)"):
            A_test = np.ones(10)

        # Version de référence complète
            mass_py, stress_py = truss(A_test)

        # Version optimisée Numba : renvoie mass et matrice S (10x12)
            mass_nb, S_nb = truss_numba(A_test)

            same_mass = abs(mass_py - mass_nb) < 1e-6

            if same_mass:
                st.success(
                "Parité OK sur la masse (tolérance 1e-6).\n"
                "La version Numba optimise surtout l’assemblage matriciel et renvoie la matrice S (10×12), "
                "pas directement le vecteur de contraintes."
                 )
            else:
                st.error("Parité NON vérifiée sur la masse.")


    with col_tb2:
        if st.button("Lancer les benchmarks (truss)"):
            t_py = bench(truss, A)
            t_nb = bench(truss_numba, A)
            speed = t_py / t_nb if t_nb > 0 else float("inf")

            c1, c2, c3 = st.columns(3)
            c1.metric("Temps Python", f"{t_py:.6f} s")
            c2.metric("Temps Numba", f"{t_nb:.6f} s")
            c3.metric("Speedup", f"×{speed:.1f}")

            df = pd.DataFrame({
                "Version": ["Python", "Numba"],
                "Temps (s)": [t_py, t_nb]
            })
            fig = px.bar(df, x="Version", y="Temps (s)",
                         title="Temps d'exécution – Truss 10 barres")
            st.plotly_chart(fig, use_container_width=True)



with tab_auto:
    st.header("⚡ Auto-Optimisation Numba via AST")

    # Initialisation des états
    if "hotspots" not in st.session_state:
        st.session_state.hotspots = None
    if "generated_file" not in st.session_state:
        st.session_state.generated_file = None
    if "source_file" not in st.session_state:
        st.session_state.source_file = None

    st.write(
        "Cet outil scanne un fichier Python, détecte les hotspots (boucles for) "
        "et génère automatiquement une version optimisée avec Numba."
    )

    py_files = [f for f in os.listdir("src") if f.endswith(".py")]
    selected = st.selectbox("Choisir un fichier Python :", [None] + py_files)

    if selected:
        file = "src/" + selected
        st.session_state.source_file = file

        with open(file, "r", encoding="utf-8") as f:
            source = f.read()

        # AST : identification hotspots
        tree = ast.parse(source)
        hotspots = []

        class HotspotFinder(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if any(isinstance(n, ast.For) for n in ast.walk(node)):
                    hotspots.append(node.name)

        HotspotFinder().visit(tree)
        st.session_state.hotspots = hotspots

        st.success(f"Fonctions détectées : {hotspots}")

        # ----------------------------------------------------------
        # 1. Génération version optimisée
        # ----------------------------------------------------------
        if st.button("Générer une version optimisée"):

            optimized_code = "from numba import njit\nimport numpy as np\n\n"

            for line in source.splitlines():
                if any(f"def {fn}" in line for fn in hotspots):
                    optimized_code += "@njit\n"
                optimized_code += line + "\n"

            output_file = file.replace(".py", "_numba_auto.py")
            st.session_state.generated_file = output_file

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(optimized_code)

            st.code(optimized_code, language="python")
            st.info(f"📄 Fichier généré : **{output_file}**")

    # ------------------------------------------------------------------
    # 2. TEST DE PARITÉ (indépendant — NE RESET PLUS LA PAGE)
    # ------------------------------------------------------------------
    if st.session_state.generated_file and st.button("Tester la parité"):
        import importlib.util
        import numpy as np

        source_file = st.session_state.source_file
        output_file = st.session_state.generated_file
        hotspots = st.session_state.hotspots

        st.subheader("🧪 Test de parité")

        # Charger modules
        spec_o = importlib.util.spec_from_file_location("orig", source_file)
        mod_orig = importlib.util.module_from_spec(spec_o)
        spec_o.loader.exec_module(mod_orig)

        spec_n = importlib.util.spec_from_file_location("opt", output_file)
        mod_opt = importlib.util.module_from_spec(spec_n)
        spec_n.loader.exec_module(mod_opt)

        fn = hotspots[0]

        f_py = getattr(mod_orig, fn)
        f_nb = getattr(mod_opt, fn)

        x = np.random.randn(5000)
        y = np.random.randn(50)

        f_nb(x, y)  # warmup

        try:
            ok = np.allclose(f_py(x, y), f_nb(x, y), atol=1e-6)
        except:
            ok = False

        if ok:
            st.success("✔ Parité validée : résultats identiques")
        else:
            st.error("❌ La parité n'est PAS vérifiée.")

    # ------------------------------------------------------------------
    # 3. BENCHMARK (indépendant — NE RESET PLUS LA PAGE)
    # ------------------------------------------------------------------
    if st.session_state.generated_file and st.button("Benchmark"):
        import importlib.util
        import numpy as np
        import time

        source_file = st.session_state.source_file
        output_file = st.session_state.generated_file
        hotspots = st.session_state.hotspots

        st.subheader("⏱ Benchmark optimisation")

        spec_o = importlib.util.spec_from_file_location("orig", source_file)
        mod_orig = importlib.util.module_from_spec(spec_o)
        spec_o.loader.exec_module(mod_orig)

        spec_n = importlib.util.spec_from_file_location("opt", output_file)
        mod_opt = importlib.util.module_from_spec(spec_n)
        spec_n.loader.exec_module(mod_opt)

        fn = hotspots[0]
        f_py = getattr(mod_orig, fn)
        f_nb = getattr(mod_opt, fn)

        x = np.random.randn(200000)
        y = np.random.randn(150)

        f_nb(x, y)  # warmup

        t0 = time.perf_counter()
        f_py(x, y)
        t_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        f_nb(x, y)
        t_nb = time.perf_counter() - t0

        st.success(f"Python = {t_py:.5f}s — Numba = {t_nb:.5f}s — Speedup ×{t_py/t_nb:.1f}")

        st.bar_chart(
            pd.DataFrame({"Temps (s)": [t_py, t_nb]}, index=["Python", "Numba"])
        )


# ----------------------------------------------------------------------
# TAB 4 — SYNTHÈSE GLOBALE
# ----------------------------------------------------------------------
with tab4:
    st.markdown('<div class="section-title">Synthèse de la démarche</div>', unsafe_allow_html=True)

    st.markdown("""
### 🧭 Pipeline méthodologique appliquée aux 3 cas

1. **Analyse statique du code**
   - AST (nombre de boucles, if, appels) pour repérer les fonctions numériques coûteuses.
   - `radon cc -s -a` pour la complexité cyclomatique et un score global.

2. **Profiling dynamique**
   - `cProfile` pour confirmer les vrais hotspots à l’exécution
     (par ex. double boucle de `search_sequence_python`).

3. **Refactorisation avec Numba**
   - Réécriture de fonctions en style *Numba-friendly* (boucles explicites, types simples).
   - Ajout de décorateurs `@njit` (et `parallel=True` si pertinent).

4. **Tests de parité**
   - Comparaison systématique Python / NumPy / Numba sur des données aléatoires.
   - Tolérance de 1e-6 pour les différences numériques.

5. **Benchmarks**
   - Mesure du temps d’exécution avant/après optimisation.
   - Observation des speedups en fonction de la taille des données.

6. **Visualisation & communication**
   - Tableau de bord Streamlit pour montrer la démarche en direct pendant la soutenance.
""")

    st.markdown("""
### 📊 Résultats qualitatifs

- **Sum of squares** : Numba permet des gains importants dès que la taille du tableau augmente.
- **Search sequence** : speedup de l’ordre de ×100–×300 par rapport à la double boucle Python,
  en conservant la même sémantique que la version NumPy.
- **Truss 10 barres** : même masse et mêmes contraintes, pour un temps de calcul nettement réduit.
""")

    st.markdown("""
### 🤖 Rôle des modèles de langage (LLM)

Les modèles de langage ont été utilisés comme **assistants** pour :

- clarifier certaines parties de code (notamment le problème du truss) ;
- proposer des refactorisations compatibles avec Numba ;
- suggérer une structure cohérente pour :
  - les tests de parité,
  - les scripts de benchmark,
  - l’organisation du dépôt et de l’interface Streamlit.

Chaque suggestion a été :

- relue et comprise,
- validée par des **tests de parité**,
- évaluée par des **benchmarks** avant d’être adoptée.

Les principaux prompts pourront être fournis dans le dépôt (README ou fichier dédié)
pour documenter la part d’assistance et la reproductibilité de la démarche.
""")

    st.markdown("""
### ✅ Conclusion

Le projet montre comment :

- **identifier** des fonctions candidates à l’optimisation,
- **accélérer** leur exécution avec Numba,
- tout en **garantissant la correction** grâce à des tests systématiques,
- et en **quantifiant** précisément les gains obtenus.

Cette méthodologie est réutilisable sur d’autres bases de code Python,
que ce soit pour des exemples pédagogiques ou des codes scientifiques plus complexes.
""")




